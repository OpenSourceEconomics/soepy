import numba
import numpy as np

from soepy.shared.shared_constants import INVALID_FLOAT
from soepy.shared.shared_constants import NUM_CHOICES
from soepy.shared.tax_and_transfers import calculate_net_income


def draw_disturbances(seed, num_periods, num_draws, model_params):
    """Creates desired number of draws of a multivariate standard normal
    distribution.

    """
    np.random.seed(seed)

    # Input parameters of the distribution
    mean = [0, 0]
    shocks_cov_matrix = np.zeros((2, 2), float)
    np.fill_diagonal(shocks_cov_matrix, model_params.shocks_cov)

    # Create draws from the standard normal distribution
    draws = np.random.multivariate_normal(
        mean, shocks_cov_matrix, (num_periods, num_draws)
    )

    return draws


def calculate_utility_components(
    model_params, model_spec, states, covariates, is_expected
):
    """Calculate utility components for all choices given state, period, and shocks.

    Parameters
    ----------
    model_params : namedtuple
        Contains all parameters of the model including information on dimensions
        (number of periods, agents, random draws, etc.) and coefficients to be
        estimated.
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    covariates: np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.
    is_expected: bool
        A boolean indicator that differentiates between the human capital accumulation
        process that agents expect (is_expected = True) and that the market generates
        (is_expected = False)

    Returns
    -------
    log_wage_systematic : array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.

    """
    if is_expected:
        # Calculate biased part-time expectation by using ratio from expected data and structural paramteters
        gamma_p = (
            model_params.gamma_p_bias / (model_params.gamma_p / model_params.gamma_f)
        ) * model_params.gamma_p
    else:
        gamma_p = model_params.gamma_p
    log_wage_systematic = calculate_log_wage_systematic(
        gamma_0=model_params.gamma_0,
        gamma_p=gamma_p,
        gamma_f=model_params.gamma_f,
        model_spec=model_spec,
        states=states,
    )

    non_consumption_utility = calculate_non_consumption_utility(
        model_params, model_spec, states, covariates
    )

    return log_wage_systematic, non_consumption_utility


def calculate_log_wage_systematic(gamma_0, gamma_f, gamma_p, model_spec, states):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p_state, exp_f_state = states[:, 3], states[:, 4]

    log_exp_p = np.log(
        np.where(
            exp_p_state + exp_f_state > model_spec.exp_cap,
            np.around(
                exp_p_state / (exp_p_state + exp_f_state + 0.5) * model_spec.exp_cap
            ),
            exp_p_state,
        )
        + 1
    )

    log_exp_f = np.log(
        np.where(
            exp_p_state + exp_f_state > model_spec.exp_cap,
            np.around(
                exp_f_state / (exp_p_state + exp_f_state + 0.5) * model_spec.exp_cap
            ),
            exp_f_state,
        )
        + 1
    )

    # Construct wage components
    gamma_0_edu = gamma_0[states[:, 1]]
    gamma_f_edu = gamma_f[states[:, 1]]
    gamma_p_edu = gamma_p[states[:, 1]]

    # Calculate wage in the given state
    log_wage_systematic = (
        gamma_0_edu + gamma_f_edu * log_exp_f + gamma_p_edu * log_exp_p
    )

    return log_wage_systematic


def calculate_non_consumption_utility(model_params, model_spec, states, covariates):
    """Calculate non-pecuniary utility contribution."""

    non_consumption_utility = np.full(
        (states.shape[0], NUM_CHOICES), [0.00] * NUM_CHOICES
    )

    # Type contribution
    # TODO: Can I get rid of the 1st zero everywhere?
    for i in range(1, model_spec.num_types):
        non_consumption_utility[np.where(states[:, 5] == i)] += [
            0,
            model_params.theta_p[i - 1],
            model_params.theta_f[i - 1],
        ]

    # Children contribution
    # No children:
    for educ_level in [0, 1, 2]:

        non_consumption_utility[
            np.where((covariates[:, 0] == 0) & (states[:, 1] == educ_level))
        ] += [
            0,  # non-employed
            model_params.no_kids_f[educ_level]
            + model_params.no_kids_p[
                educ_level
            ],  # part-time alpha_f_no_kids + alpha_p_no_kids
            model_params.no_kids_f[educ_level],  # full-time alpha_f_no_kids
        ]

        # Children present:
        non_consumption_utility[
            np.where((covariates[:, 0] != 0) & (states[:, 1] == educ_level))
        ] += [
            0,
            model_params.yes_kids_f[educ_level] + model_params.yes_kids_p[educ_level],
            model_params.yes_kids_f[educ_level],
        ]

    # Contribution child aged 0-2:
    non_consumption_utility[np.where(covariates[:, 0] == 1)] += [
        0,
        model_params.child_02_f + model_params.child_02_p,
        model_params.child_02_f,
    ]

    # Contribution child aged 3-5:
    non_consumption_utility[np.where(covariates[:, 0] == 2)] += [
        0,
        model_params.child_35_f + model_params.child_35_p,
        model_params.child_35_f,
    ]

    # Contribution child aged 6-10:
    non_consumption_utility[np.where(covariates[:, 0] == 3)] += [
        0,
        model_params.child_6orolder_f + model_params.child_6orolder_p,
        model_params.child_6orolder_f,
    ]

    non_consumption_utility = np.exp(non_consumption_utility)

    return non_consumption_utility


@numba.jit(nopython=True)
def calculate_non_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    male_wage,
    non_employment_benefits,
    tax_splitting=True,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to not be employed.
    It adds the components from the budget constraint to the female wage."""

    non_employment_consumption_resources = np.full(male_wage.shape[0], INVALID_FLOAT)

    for i in range(male_wage.shape[0]):
        # Set female wage to
        net_income = (
            calculate_net_income(
                income_tax_spec, deductions_spec, 0, male_wage[i], tax_splitting
            )
            + non_employment_benefits[i, 0]
        )

        non_employment_consumption_resources[i] = (
            net_income + non_employment_benefits[i, 1] + non_employment_benefits[i, 2]
        )

    return non_employment_consumption_resources


@numba.jit(nopython=True)
def calculate_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    current_female_income,
    male_wage,
    tax_splitting=True,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to be employed.
    It adds the components from the budget constraint to the female wage."""

    employment_consumption_resources = np.full(
        current_female_income.shape, INVALID_FLOAT
    )

    for i in range(current_female_income.shape[0]):
        male_wage_i = male_wage[i]
        for choice_num in range(current_female_income.shape[1]):
            employment_consumption_resources[i, choice_num] = calculate_net_income(
                income_tax_spec,
                deductions_spec,
                current_female_income[i, choice_num],
                male_wage_i,
                tax_splitting,
            )

    return employment_consumption_resources
