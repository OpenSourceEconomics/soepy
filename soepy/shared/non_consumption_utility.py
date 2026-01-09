import jax.numpy as jnp


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
    educ = states[:, 1]
    unobs_types = states[:, 5]

    # Base utilities
    util_pt = model_params.theta_p[unobs_types]
    util_ft = model_params.theta_f[unobs_types]

    # Binary selectors (implicitly cast to 0/1 in arithmetic)
    b0 = child_bins == 0
    b1 = child_bins == 1
    b2 = child_bins == 2
    b3 = child_bins == 3
    b4 = child_bins > 3

    # Education-dependent components
    no_kids_f = model_params.no_kids_f[educ]
    no_kids_p = model_params.no_kids_p[educ]
    yes_kids_f = model_params.yes_kids_f[educ]
    yes_kids_p = model_params.yes_kids_p[educ]

    # Part-time utility (always includes full time base utility)
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

    # Full-time utility
    util_ft += (
        b0 * no_kids_f
        + b1 * (yes_kids_f + model_params.child_0_2_f)
        + b2 * (yes_kids_f + model_params.child_3_5_f)
        + b3 * (yes_kids_f + model_params.child_6_10_f)
        + b4 * yes_kids_f
    )

    # Stack: [no work, part-time, full-time]
    non_consumption_utility = jnp.stack(
        (
            jnp.zeros_like(util_pt),
            util_pt,
            util_ft,
        ),
        axis=1,
    )

    return jnp.exp(non_consumption_utility)
