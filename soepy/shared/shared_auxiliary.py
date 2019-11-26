import numpy as np

from soepy.shared.shared_constants import NUM_CHOICES


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
        model_params, model_spec, states
    )

    return log_wage_systematic, non_consumption_utility


def calculate_log_wage_systematic(model_params, states, covariates, is_expected):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p, exp_f = states[:, 3], states[:, 4]
    educ_level = covariates

    # Construct wage components
    gamma_0s = np.array(model_params.gamma_0s)[educ_level]
    gamma_1s = np.array(model_params.gamma_1s)[educ_level]

    if is_expected:
        period_exp_sum = 0.5 * exp_p + exp_f
    else:
        period_exp_sum = exp_p * np.array(model_params.g_s)[educ_level] + exp_f

    depreciation = 1 - np.array(model_params.delta_s)[educ_level]

    # Calculate wage in the given state
    period_exp_total = period_exp_sum * depreciation + 1
    returns_to_exp = gamma_1s * np.log(period_exp_total)
    log_wage_systematic = gamma_0s + returns_to_exp

    return log_wage_systematic


def calculate_non_consumption_utility(model_params, model_spec, states):
    """Calculate non-pecuniary utility contribution."""

    non_consumption_utility = np.full(
        (states.shape[0], NUM_CHOICES), [0, model_params.const_p, model_params.const_f]
    )

    for i in range(1, model_spec.num_types):
        non_consumption_utility[np.where(states[:, 5] == i)] += [
            0,
            model_params.theta_p[i - 1],
            model_params.theta_f[i - 1],
        ]

    non_consumption_utility = np.exp(non_consumption_utility)

    return non_consumption_utility
