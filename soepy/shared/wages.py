from jax import numpy as jnp

from soepy.shared.experience_stock import stock_to_exp_years


def calculate_log_wage(
    model_params, educ, exp_stock, init_exp_max, pt_increment, period
):
    """Calculate systematic log wages for continuous experience.

    The continuous-experience model uses a single return to experience. Expectation
    bias is handled in the experience law of motion (pt increment), not in wages.

    Parameters
    ----------
    model_params : namedtuple
        Requires ``gamma_0`` and ``gamma_f`` (used as the single experience return).
    educ : int
        Education level index.
    exp_stock : jax.numpy.ndarray
        Experience stock between 0 and 1.
    init_exp_max : float
        Initial maximum experience years.
    pt_increment : float
        Part-time experience increment.
    period : int
        Current period.

    Returns
    -------
    jax.numpy.ndarray
        Systematic log wages with shape (n_states, n_grid).
    """

    exp_years = stock_to_exp_years(
        stock=exp_stock,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )
    gamma_0_edu = model_params.gamma_0[educ]
    gamma_exp_edu = model_params.gamma_1[educ]

    log_exp = jnp.log(exp_years + 1)
    return gamma_0_edu + gamma_exp_edu * log_exp
