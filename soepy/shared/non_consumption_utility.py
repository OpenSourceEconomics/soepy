import jax.numpy as jnp


def calculate_non_consumption_utility(model_params, educ, unobs_type, child_bin):
    """Calculate non-pecuniary utility contribution.

    Parameters
    ----------
    states : np.ndarray
        Shape (n_states, n_state_vars) matrix of discrete states.
    child_bin : np.ndarray
        Shape (n_states,) array with child bin indices for each state.

    Returns
    -------
    jax.numpy.ndarray
        Shape (n_states, 3) matrix with utilities for [no work, part-time, full-time].
    """
    util_pt = model_params.theta_p[unobs_type]
    util_ft = model_params.theta_f[unobs_type]

    b0 = child_bin == 0
    b1 = child_bin == 1
    b2 = child_bin == 2
    b3 = child_bin == 3
    b4 = child_bin > 3

    no_kids_f = model_params.no_kids_f[educ]
    no_kids_p = model_params.no_kids_p[educ]
    yes_kids_f = model_params.yes_kids_f[educ]
    yes_kids_p = model_params.yes_kids_p[educ]

    util_pt_b0 = no_kids_f + no_kids_p
    util_pt_b1 = (
        yes_kids_f + yes_kids_p + model_params.child_0_2_f + model_params.child_0_2_p
    )
    util_pt_b2 = (
        yes_kids_f + yes_kids_p + model_params.child_3_5_f + model_params.child_3_5_p
    )
    util_pt_b3 = (
        yes_kids_f + yes_kids_p + model_params.child_6_10_f + model_params.child_6_10_p
    )
    util_pt_b4 = (
        yes_kids_f
        + yes_kids_p
        + model_params.child_11_age_max_f
        + model_params.child_11_age_max_p
    )

    util_pt += (
        b0 * util_pt_b0
        + b1 * util_pt_b1
        + b2 * util_pt_b2
        + b3 * util_pt_b3
        + b4 * util_pt_b4
    )

    util_ft_b0 = no_kids_f
    util_ft_b1 = yes_kids_f + model_params.child_0_2_f
    util_ft_b2 = yes_kids_f + model_params.child_3_5_f
    util_ft_b3 = yes_kids_f + model_params.child_6_10_f
    util_ft_b4 = yes_kids_f + model_params.child_11_age_max_f

    util_ft += (
        b0 * util_ft_b0
        + b1 * util_ft_b1
        + b2 * util_ft_b2
        + b3 * util_ft_b3
        + b4 * util_ft_b4
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
