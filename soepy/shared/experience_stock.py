import jax.numpy as jnp


def get_pt_increment(model_params, educ_level, child_age, biased_exp):
    """Return the part-time experience increment.

    Rules
    -----
    - If ``biased_exp`` is True: always return 1.0 (same as full-time).
    - Else: return ``gamma_p[educ_level]`` plus ``gamma_p_mom`` if the agent's
      youngest child aged 0–2.

    Parameters
    ----------
    model_params : namedtuple-like
        Model parameters with attributes ``gamma_p`` and ``gamma_p_mom``.
    educ_level : int | jnp.ndarray
        Education level index/indices.
    child_age : int | jnp.ndarray
        Age of youngest child (use -1 for no child).
    biased_exp : bool
        Whether to use the expected law of motion.

    Returns
    -------
    float or jnp.ndarray
        Part-time increment(s).
    """

    if biased_exp:
        # Must return an array when called from JAX-jitted code.
        return jnp.ones_like(educ_level, dtype=float)

    base = model_params.gamma_p[educ_level]
    little_child = (child_age >= 0) & (child_age <= 2)

    return base + little_child * model_params.gamma_p_mom


def max_exp_years(period, init_exp_max, pt_increment):
    """Compute the maximum feasible experience years in a given period."""
    return init_exp_max + jnp.maximum(period, period * pt_increment)


def stock_to_exp_years(stock, period, init_exp_max, pt_increment):
    """Map experience stock in [0,1] to experience measured in years."""

    max_years = max_exp_years(
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )
    return stock * max_years


def exp_years_to_stock(exp_years, period, init_exp_max, pt_increment):
    """Map experience measured in years to stock in [0,1]."""

    max_years = max_exp_years(
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )

    # Avoid division by zero for corner cases (e.g. init_exp_max == 0 and period == 0).
    safe_denom = jnp.where(max_years > 0, max_years, 1.0)
    return exp_years / safe_denom


def next_exp_years(exp_years, choice, pt_increment):
    """Apply the (choice-dependent) experience accumulation in years."""

    inc_pt = (choice == 1) * pt_increment
    inc_ft = (choice == 2) * 1.0
    return exp_years + inc_pt + inc_ft


def next_stock(stock, period, init_exp_max, pt_increment, choice):
    """Transition the experience stock to the next period.

    The stock is interpreted in period-``t`` units, mapped to years, incremented based
    on the choice, and mapped back to stock in period-``t+1`` units.
    """

    exp_years = stock_to_exp_years(
        stock=stock,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )

    exp_years_next = next_exp_years(
        exp_years=exp_years,
        choice=choice,
        pt_increment=pt_increment,
    )

    stock_next = exp_years_to_stock(
        exp_years=exp_years_next,
        period=period + 1,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )

    return jnp.clip(stock_next, 0.0, 1.0)
