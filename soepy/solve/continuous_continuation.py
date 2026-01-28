"""Continuation value computation for continuous experience.

Ordering requirement:

1. interpolate child-state value functions on the experience grid
2. aggregate over child and partner probabilities

This module is written for readability. It is intended to be used with ``jax.vmap``
from the caller.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.experience_stock import next_stock
from soepy.shared.interpolation import linear_interp_1d


def interpolate_then_weight_continuation_values(
    exp_grid,
    v_next_grid,
    child_state_indexes_local,
    period,
    init_exp_max,
    pt_increment_states,
    prob_child_states,
    prob_partner_states,
):
    """Compute expected continuation values on the current experience grid.

    Parameters
    ----------
    exp_grid : jax.numpy.ndarray, shape (n_grid,)
        Current-period experience stock grid.
    v_next_grid : jax.numpy.ndarray, shape (n_next_states, n_grid)
        Next-period maximum value function on the same grid.
    child_state_indexes_local : jax.numpy.ndarray, shape (n_states, NUM_CHOICES, 2, 2)
        Next-period discrete state indices (local to next-period block).
    period : scalar
        Current period.
    init_exp_max : scalar
        Maximum initial experience.
    pt_increment_states : jax.numpy.ndarray, shape (n_states,)
        Part-time increment by state.
    prob_child_states : jax.numpy.ndarray, shape (n_states,)
        Child arrival probability by state.
    prob_partner_states : jax.numpy.ndarray, shape (n_states, 2)
        Partner transition probabilities by state, ordered as [p0, p1].

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_states, NUM_CHOICES, n_grid) with expected continuation values.
    """

    def one_choice(
        values_next, idx_choice, pt_increment, prob_child, prob_partner, choice
    ):
        x_next = next_stock(
            stock=exp_grid,
            period=period,
            init_exp_max=init_exp_max,
            pt_increment=pt_increment,
            choice=choice,
        )

        idx_no_child_single = idx_choice[0, 0]
        idx_no_child_partner = idx_choice[0, 1]
        idx_child_single = idx_choice[1, 0]
        idx_child_partner = idx_choice[1, 1]

        val_no_child_single = linear_interp_1d(
            exp_grid, values_next[idx_no_child_single], x_next
        )
        val_no_child_partner = linear_interp_1d(
            exp_grid, values_next[idx_no_child_partner], x_next
        )
        val_child_single = linear_interp_1d(
            exp_grid, values_next[idx_child_single], x_next
        )
        val_child_partner = linear_interp_1d(
            exp_grid, values_next[idx_child_partner], x_next
        )

        prob_no_child = 1.0 - prob_child
        prob_single, prob_partner = prob_partner[0], prob_partner[1]

        return prob_no_child * (
            prob_single * val_no_child_single + prob_partner * val_no_child_partner
        ) + prob_child * (
            prob_single * val_child_single + prob_partner * val_child_partner
        )

    def one_state(idx_state, pt_increment, prob_child, prob_partner):
        idx_choices = child_state_indexes_local[idx_state]
        choice_ids = jnp.arange(NUM_CHOICES)

        return jax.vmap(
            lambda choice: one_choice(
                values_next=v_next_grid,
                idx_choice=idx_choices[choice],
                pt_increment=pt_increment,
                prob_child=prob_child,
                prob_partner=prob_partner,
                choice=choice,
            )
        )(choice_ids)

    n_states = child_state_indexes_local.shape[0]
    state_ids = jnp.arange(n_states)

    return jax.vmap(one_state)(
        state_ids,
        pt_increment_states,
        prob_child_states,
        prob_partner_states,
    )
