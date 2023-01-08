import numba
import numpy as np

from soepy.shared.tax_and_transfers import calculate_net_income


def draw_disturbances(seed, num_periods, num_draws, model_params):
    """Creates desired number of draws of a multivariate standard normal
    distribution.

    """
    np.random.seed(seed)

    mean = 0

    # Create draws from the standard normal distribution
    draws = np.random.normal(mean, model_params.shock_sd, (num_periods, num_draws))

    return draws


def calculate_log_wage(model_params, states, is_expected):
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
        # Calculate biased part-time expectation by using ratio from expected data and
        # structural paramteters
        gamma_p = (
            model_params.gamma_p_bias / (model_params.gamma_p / model_params.gamma_f)
        ) * model_params.gamma_p
    else:
        gamma_p = model_params.gamma_p
    log_wage_systematic = calculate_log_wage_systematic(
        model_params.gamma_0,
        model_params.gamma_f,
        gamma_p,
        states,
    )

    return log_wage_systematic


@numba.guvectorize(
    ["f8[:],f8[:],f8[:],i8[:], f8[:]"],
    "(num_edu_types),(num_edu_types),(num_edu_types),(num_state_vars)->()",
    nopython=True,
    target="cpu",
    # target="parallel",
)
def calculate_log_wage_systematic(
    gamma_0, gamma_f, gamma_p, state, log_wage_systematic
):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p_state, exp_f_state = state[3], state[4]

    log_exp_p = np.log(exp_p_state + 1)
    log_exp_f = np.log(exp_f_state + 1)

    # Assign wage returns
    gamma_0_edu = gamma_0[state[1]]
    gamma_f_edu = gamma_f[state[1]]
    gamma_p_edu = gamma_p[state[1]]

    # Calculate wage in the given state
    log_wage_systematic[0] = (
        gamma_0_edu + gamma_f_edu * log_exp_f + gamma_p_edu * log_exp_p
    )


@numba.guvectorize(
    ["f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8,f8,f8,f8,f8,f8,i8[:], f8, f8[:], f8[:]"],
    "(num_unobs_types),(num_unobs_types), (num_edu_types),(num_edu_types),"
    "(num_edu_types),(num_edu_types), (),(),(),(),(),(),(num_state_vars),"
    "(),(num_choices)->(num_choices)",
    nopython=True,
    target="cpu",
    # target="parallel",
)
def calculate_non_consumption_utility(
    theta_p,
    theta_f,
    no_kids_f,
    no_kids_p,
    yes_kids_f,
    yes_kids_p,
    child_0_2_f,
    child_0_2_p,
    child_3_5_f,
    child_3_5_p,
    child_6_10_f,
    child_6_10_p,
    state,
    child_bin,
    dummy_output,
    non_consumption_utility,
):
    """Calculate non-pecuniary utility contribution."""
    non_consumption_utility_state = np.array([0, theta_p[state[5]], theta_f[state[5]]])
    educ_level = state[1]
    if child_bin == 0:
        non_consumption_utility_state[1] += (
            no_kids_f[educ_level] + no_kids_p[educ_level]
        )  # part-time alpha_f_no_kids + alpha_p_no_kids
        non_consumption_utility_state[2] += no_kids_f[educ_level]
    elif child_bin == 1:
        non_consumption_utility_state[1] += (
            yes_kids_f[educ_level] + yes_kids_p[educ_level] + child_0_2_f + child_0_2_p
        )

        non_consumption_utility_state[2] += yes_kids_f[educ_level] + child_0_2_f
    elif child_bin == 2:
        non_consumption_utility_state[1] += (
            yes_kids_f[educ_level] + yes_kids_p[educ_level] + child_3_5_f + child_3_5_p
        )
        non_consumption_utility_state[2] += yes_kids_f[educ_level] + child_3_5_f

    elif child_bin == 3:
        non_consumption_utility_state[1] += (
            yes_kids_f[educ_level]
            + yes_kids_p[educ_level]
            + child_6_10_f
            + child_6_10_p
        )
        non_consumption_utility_state[2] += yes_kids_f[educ_level] + child_6_10_f
    else:  # Mothers with kids older than 10 only get fixed disutility
        non_consumption_utility_state[1] += (
            yes_kids_f[educ_level] + yes_kids_p[educ_level]
        )

        non_consumption_utility_state[2] += yes_kids_f[educ_level]
    out = np.exp(non_consumption_utility_state)
    non_consumption_utility[0] = out[0]
    non_consumption_utility[1] = out[1]
    non_consumption_utility[2] = out[2]


@numba.guvectorize(
    ["f8[:], f8[:, :], f8[:], f8, b1, f8[:]"],
    "(n_ssc_params), (n_tax_params, n_tax_params), (num_work_choices), (), () -> (num_work_choices)",
    nopython=True,
    target="cpu",
    # target="parallel",
)
def calculate_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    current_female_income,
    male_wage,
    tax_splitting,
    employment_consumption_resources,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to be employed.
    It adds the components from the budget constraint to the female wage."""

    for choice_num in range(current_female_income.shape[0]):
        employment_consumption_resources[choice_num] = calculate_net_income(
            income_tax_spec,
            deductions_spec,
            current_female_income[choice_num],
            male_wage,
            tax_splitting,
        )
