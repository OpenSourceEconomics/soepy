"""Experience stock utilities.

This module defines a continuous experience stock ``x`` on [0, 1]. Internally, we map
between the stock and experience measured in (model) years.

The period-specific scale is

    max_exp_years(period) = init_exp_max + max(period, period * pt_increment)

where ``pt_increment`` is the per-period increment in experience from part-time work
(full-time increments by 1).

The functions are implemented with ``jax.numpy`` so they can be used in jitted code.
"""
from __future__ import annotations

import jax.numpy as jnp


def get_pt_increment(model_params, educ_level, is_expected):
    """Return the part-time experience increment.

    Parameters
    ----------
    model_params : namedtuple-like
        Model parameters with attribute ``gamma_p`` or ``gamma_p_bias``.
    is_expected : bool
        Whether to use the expected law of motion.
    educ_level : int | None
        Optional education level index when the increment is education-specific.

    Returns
    -------
    float or jnp.ndarray
        The part-time increment.
    """

    if is_expected:
        increment = model_params.gamma_p_bias
    else:
        increment = model_params.gamma_p

    return increment[educ_level]


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
