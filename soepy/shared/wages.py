from jax import numpy as jnp

from soepy.shared.state_space_indices import EDUC_LEVEL


def calculate_log_wage(model_params, states, exp_years):
    """Calculate systematic log wages for continuous experience.

    The continuous-experience model uses a single return to experience. Expectation
    bias is handled in the experience law of motion (pt increment), not in wages.

    Parameters
    ----------
    model_params : namedtuple
        Requires ``gamma_0`` and ``gamma_f`` (used as the single experience return).
    states : jax.numpy.ndarray
        Discrete state array with education level in column ``EDUC_LEVEL``.
    exp_years : jax.numpy.ndarray
        Experience measured in years. Shape (n_states, n_grid).

    Returns
    -------
    jax.numpy.ndarray
        Systematic log wages with shape (n_states, n_grid).
    """

    educ = states[:, EDUC_LEVEL]

    gamma_0_edu = model_params.gamma_0[educ]
    gamma_exp_edu = model_params.gamma_f[educ]

    log_exp = jnp.log(exp_years + 1)
    return gamma_0_edu[:, None] + gamma_exp_edu[:, None] * log_exp
