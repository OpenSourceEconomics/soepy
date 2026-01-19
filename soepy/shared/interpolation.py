"""Interpolation helpers.

This module contains small JAX-friendly interpolation primitives used during solution.

Note: We intentionally keep this minimal. The main use case is to ``vmap`` this
function over states and/or query points.
"""
from __future__ import annotations

import jax.numpy as jnp


def linear_interp_1d(grid, values, x):
    """Linearly interpolate 1D values defined on a 1D grid.

    Parameters
    ----------
    grid : array-like, shape (n_grid,)
        Increasing grid.
    values : array-like, shape (n_grid,)
        Values defined on the grid.
    x : array-like
        Query point(s).

    Returns
    -------
    jax.numpy.ndarray
        Interpolated value(s) with the same shape as ``x``.
    """

    idx_hi = jnp.searchsorted(grid, x, side="right")
    idx_hi = jnp.clip(idx_hi, 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1

    x_lo = grid[idx_lo]
    x_hi = grid[idx_hi]

    v_lo = values[idx_lo]
    v_hi = values[idx_hi]

    denom = x_hi - x_lo
    zero = jnp.zeros_like(x)
    weight = jnp.where(denom > 0, (x - x_lo) / denom, zero)

    return v_lo + weight * (v_hi - v_lo)
