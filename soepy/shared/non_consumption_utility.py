import jax.numpy as jnp

from soepy.shared.state_space_indices import EDUC_LEVEL
from soepy.shared.state_space_indices import TYPE


def calculate_non_consumption_utility(model_params, states, child_bins):
    """Calculate non-pecuniary utility contribution.

    Parameters
    ----------
    states : np.ndarray
        Shape (n_states, n_state_vars) matrix of discrete states.
    child_bins : np.ndarray
        Shape (n_states,) array with child bin indices for each state.

    Returns
    -------
    jax.numpy.ndarray
        Shape (n_states, 3) matrix with utilities for [no work, part-time, full-time].
    """

    educ = states[:, EDUC_LEVEL]
    unobs_types = states[:, TYPE]

    util_pt = model_params.theta_p[unobs_types]
    util_ft = model_params.theta_f[unobs_types]

    b0 = child_bins == 0
    b1 = child_bins == 1
    b2 = child_bins == 2
    b3 = child_bins == 3
    b4 = child_bins > 3

    no_kids_f = model_params.no_kids_f[educ]
    no_kids_p = model_params.no_kids_p[educ]
    yes_kids_f = model_params.yes_kids_f[educ]
    yes_kids_p = model_params.yes_kids_p[educ]

    util_pt += (
        b0 * (no_kids_f + no_kids_p)
        + b1
        * (
            yes_kids_f
            + yes_kids_p
            + model_params.child_0_2_f
            + model_params.child_0_2_p
        )
        + b2
        * (
            yes_kids_f
            + yes_kids_p
            + model_params.child_3_5_f
            + model_params.child_3_5_p
        )
        + b3
        * (
            yes_kids_f
            + yes_kids_p
            + model_params.child_6_10_f
            + model_params.child_6_10_p
        )
        + b4 * (yes_kids_f + yes_kids_p)
    )

    util_ft += (
        b0 * no_kids_f
        + b1 * (yes_kids_f + model_params.child_0_2_f)
        + b2 * (yes_kids_f + model_params.child_3_5_f)
        + b3 * (yes_kids_f + model_params.child_6_10_f)
        + b4 * yes_kids_f
    )

    non_consumption_utility = jnp.stack(
        (
            jnp.zeros_like(util_pt),
            util_pt,
            util_ft,
        ),
        axis=1,
    )

    return jnp.exp(non_consumption_utility)
