import numpy as np
from jax import numpy as jnp


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
    gamma_0_edu = jnp.array(gamma_0)[states[:, 1]]
    gamma_f_edu = jnp.array(gamma_f)[states[:, 1]]
    gamma_p_edu = jnp.array(gamma_p)[states[:, 1]]

    # Calculate wage in the given state
    log_wage_systematic = (
        gamma_0_edu + gamma_f_edu * log_exp_f + gamma_p_edu * log_exp_p
    )
    return log_wage_systematic
