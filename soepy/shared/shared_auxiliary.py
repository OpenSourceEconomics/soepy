import jax.numpy as jnp
import numpy as np


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


def calculate_log_wage_systematic(gamma_0, gamma_f, gamma_p, states):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p_states, exp_f_states = states[:, 3], states[:, 4]

    log_exp_p = jnp.log(exp_p_states + 1)
    log_exp_f = jnp.log(exp_f_states + 1)

    # Assign wage returns
    gamma_0_edu = gamma_0[states[:, 1]]
    gamma_f_edu = gamma_f[states[:, 1]]
    gamma_p_edu = gamma_p[states[:, 1]]

    # Calculate wage in the given state
    log_wage_systematic = (
        gamma_0_edu + gamma_f_edu * log_exp_f + gamma_p_edu * log_exp_p
    )
    return log_wage_systematic


def calculate_non_consumption_utility(
    model_params,
    states,
    child_bins,
):
    """Calculate non-pecuniary utility contribution.

    Parameters
    ----------
    states : np.ndarray
        Shape (n_states, n_state_vars) matrix of states
    child_bins : np.ndarray
        Shape (n_states,) array with child bin indices for each state

    Returns
    -------
    non_consumption_utility : np.ndarray
        Shape (n_states, 3) matrix with utilities for [no work, part-time, full-time]
    """
    theta_p = model_params.theta_p
    theta_f = model_params.theta_f
    no_kids_f = model_params.no_kids_f
    no_kids_p = model_params.no_kids_p
    yes_kids_f = model_params.yes_kids_f
    yes_kids_p = model_params.yes_kids_p
    child_0_2_f = model_params.child_0_2_f
    child_0_2_p = model_params.child_0_2_p
    child_3_5_f = model_params.child_3_5_f
    child_3_5_p = model_params.child_3_5_p
    child_6_10_f = model_params.child_6_10_f
    child_6_10_p = model_params.child_6_10_p

    n_states = states.shape[0]
    educ_levels = states[:, 1]  # Extract education level for all states

    # Initialize output: column 0 (no work) = 0, columns 1-2 get theta values
    non_consumption_utility = np.zeros((n_states, 3))
    non_consumption_utility[:, 1] = theta_p[states[:, 5]]  # part-time base utility
    non_consumption_utility[:, 2] = theta_f[states[:, 5]]  # full-time base utility

    # Create masks for each child bin
    no_kids_mask = child_bins == 0
    child_0_2_mask = child_bins == 1
    child_3_5_mask = child_bins == 2
    child_6_10_mask = child_bins == 3
    older_kids_mask = child_bins > 3

    # No kids (child_bin == 0)
    non_consumption_utility[no_kids_mask, 1] += (
        no_kids_f[educ_levels[no_kids_mask]] + no_kids_p[educ_levels[no_kids_mask]]
    )
    non_consumption_utility[no_kids_mask, 2] += no_kids_f[educ_levels[no_kids_mask]]

    # Child 0-2 (child_bin == 1)
    non_consumption_utility[child_0_2_mask, 1] += (
        yes_kids_f[educ_levels[child_0_2_mask]]
        + yes_kids_p[educ_levels[child_0_2_mask]]
        + child_0_2_f
        + child_0_2_p
    )
    non_consumption_utility[child_0_2_mask, 2] += (
        yes_kids_f[educ_levels[child_0_2_mask]] + child_0_2_f
    )

    # Child 3-5 (child_bin == 2)
    non_consumption_utility[child_3_5_mask, 1] += (
        yes_kids_f[educ_levels[child_3_5_mask]]
        + yes_kids_p[educ_levels[child_3_5_mask]]
        + child_3_5_f
        + child_3_5_p
    )
    non_consumption_utility[child_3_5_mask, 2] += (
        yes_kids_f[educ_levels[child_3_5_mask]] + child_3_5_f
    )

    # Child 6-10 (child_bin == 3)
    non_consumption_utility[child_6_10_mask, 1] += (
        yes_kids_f[educ_levels[child_6_10_mask]]
        + yes_kids_p[educ_levels[child_6_10_mask]]
        + child_6_10_f
        + child_6_10_p
    )
    non_consumption_utility[child_6_10_mask, 2] += (
        yes_kids_f[educ_levels[child_6_10_mask]] + child_6_10_f
    )

    # Older kids (child_bin > 3)
    non_consumption_utility[older_kids_mask, 1] += (
        yes_kids_f[educ_levels[older_kids_mask]]
        + yes_kids_p[educ_levels[older_kids_mask]]
    )
    non_consumption_utility[older_kids_mask, 2] += yes_kids_f[
        educ_levels[older_kids_mask]
    ]

    # Apply exponential transformation
    non_consumption_utility = np.exp(non_consumption_utility)

    return non_consumption_utility
