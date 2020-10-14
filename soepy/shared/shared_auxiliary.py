import numpy as np

from soepy.shared.shared_constants import NUM_CHOICES, HOURS


def draw_disturbances(seed, num_periods, num_draws, model_params):
    """Creates desired number of draws of a multivariate standard normal
    distribution.

    """
    np.random.seed(seed)

    # Input parameters of the distribution
    mean = [0, 0, 0]
    shocks_cov_matrix = np.zeros((3, 3), float)
    np.fill_diagonal(shocks_cov_matrix, model_params.shocks_cov)

    # Create draws from the standard normal distribution
    draws = np.random.multivariate_normal(
        mean, shocks_cov_matrix, (num_periods, num_draws)
    )

    draws = draws[:, :, 1:]

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
    log_wage_systematic = calculate_log_wage_systematic(
        model_params, states, covariates, is_expected
    )

    non_consumption_utility = calculate_non_consumption_utility(
        model_params, model_spec, states, covariates
    )

    return log_wage_systematic, non_consumption_utility


def calculate_log_wage_systematic(model_params, states, covariates, is_expected):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p, exp_f = states[:, 3], states[:, 4]

    # Construct wage components
    gamma_0s = np.array(model_params.gamma_0s)[states[:, 1]]
    gamma_1s = np.array(model_params.gamma_1s)[states[:, 1]]

    if is_expected:
        period_exp_sum = 0.5 * exp_p + exp_f
    else:
        period_exp_sum = exp_p * np.array(model_params.g_s)[states[:, 1]] + exp_f

    depreciation = 1 - np.array(model_params.delta_s)[states[:, 1]]

    # Calculate wage in the given state
    period_exp_total = period_exp_sum * depreciation + 1
    returns_to_exp = gamma_1s * np.log(period_exp_total)
    log_wage_systematic = gamma_0s + returns_to_exp

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
    # No children
    non_consumption_utility[np.where(covariates[:, 0] == 0)] += [
        0,  # non-employed
        model_params.no_kids_f
        + model_params.no_kids_p,  # part-time alpha_f_no_kids + alpha_p_no_kids
        model_params.no_kids_f,  # full-time alpha_f_no_kids
    ]

    # Children present:
    non_consumption_utility[np.where(covariates[:, 0] != 0)] += [
        0,
        model_params.yes_kids_f + model_params.yes_kids_p,
        model_params.yes_kids_f,
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
        model_params.child_610_f + model_params.child_610_p,
        model_params.child_610_f,
    ]

    non_consumption_utility = np.exp(non_consumption_utility)

    return non_consumption_utility


def calculate_non_employment_benefits(model_spec, states, log_wage_systematic):
    """This function calculates the benefits an individual would receive if they were
    to choose to be non-employed in the period"""

    non_employment_benefits = np.full(states.shape[0], np.nan)

    # benefits base per week for a person who did not work last period
    non_employment_benefits = np.where(
        states[:, 2] == 0, model_spec.benefits_base, non_employment_benefits
    )

    # Half the labor income the individual would have earned
    # working full-time in the period (excluding wage shock)
    # for a person who worked last period
    non_employment_benefits = np.where(
        states[:, 2] != 0,
        0.5 * np.exp(log_wage_systematic) * HOURS[2],
        non_employment_benefits,
    )

    # Make sure that every state has been assigns a corresponding value
    # of non-employment benefits
    assert np.isfinite(non_employment_benefits).all()

    return non_employment_benefits


def calculate_budget_constraint_components(model_spec, states, covariates):
    """This function calculates the resources available to the woman to spend on consumption.
    It adds the components from the budget constraint to the female wage."""

    # Male wages
    budget_constraint_components = covariates[:, 1]

    # Childcare benefits
    # Benefits kids added if the person has a child
    budget_constraint_components = np.where(
        states[:, 6] != -1,
        budget_constraint_components + model_spec.benefits_kids,
        budget_constraint_components,
    )

    return budget_constraint_components
