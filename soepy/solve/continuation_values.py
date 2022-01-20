import numba


@numba.njit(nogil=True)
def get_continuation_values(
    states_subset,
    indexer,
    emaxs,
    child_age_update_rule,
    prob_child_period,
    prob_partner_process,
):
    """Obtain continuation values for each of the choices at each state
    of the period currently reached by the parent loop.

    This function takes a parent node and looks up the continuation values
    of each of the available choices. It takes the entire block of
    data:`emaxs` corresponding to the period and fills in the first block
    of elements corresponding to the continuation values.
    The continuation value of each choice is the expected maximum value
    function of the next period's state if the particular choice was
    taken this period. The expected maximum value function values are
    contained as the last element of the data:`emaxs` row of next
    period's state.

    Warning
    -------
    This function must be extremely performant as the lookup is done for each state in a
    state space (except for states in the last period) for each evaluation of the
    optimization of parameters.
    """
    for i in range(states_subset.shape[0]):

        # Unpack parent state and get index
        (
            period,
            educ_level,
            choice_lagged,
            exp_p,
            exp_f,
            type_,
            age_kid,
            partner_indicator,
        ) = states_subset[i]

        k_parent = indexer[
            period,
            educ_level,
            choice_lagged,
            exp_p,
            exp_f,
            type_,
            age_kid,
            partner_indicator,
        ]

        # Child: No arrival
        # Choice: Non-employment
        k_0_00 = indexer[
            period + 1,
            educ_level,
            0,
            exp_p,
            exp_f,
            type_,
            child_age_update_rule[k_parent],
            0,  # No partner
        ]

        k_0_01 = indexer[
            period + 1,
            educ_level,
            0,
            exp_p,
            exp_f,
            type_,
            child_age_update_rule[k_parent],
            1,  # Partner
        ]

        # Choice: Part-time
        k_1_00 = indexer[
            period + 1,
            educ_level,
            1,
            exp_p + 1,
            exp_f,
            type_,
            child_age_update_rule[k_parent],
            0,  # No partner
        ]

        k_1_01 = indexer[
            period + 1,
            educ_level,
            1,
            exp_p + 1,
            exp_f,
            type_,
            child_age_update_rule[k_parent],
            1,  # Partner
        ]

        # Choice: Full-time
        k_2_00 = indexer[
            period + 1,
            educ_level,
            2,
            exp_p,
            exp_f + 1,
            type_,
            child_age_update_rule[k_parent],
            0,  # No partner
        ]

        k_2_01 = indexer[
            period + 1,
            educ_level,
            2,
            exp_p,
            exp_f + 1,
            type_,
            child_age_update_rule[k_parent],
            1,  # Partner
        ]

        # Child possible, integrate out partner and child probability
        k_0_10 = indexer[
            period + 1, educ_level, 0, exp_p, exp_f, type_, 0, 0,  # No partner
        ]

        k_0_11 = indexer[
            period + 1, educ_level, 0, exp_p, exp_f, type_, 0, 1,  # Partner
        ]

        # Choice: Part-time
        k_1_10 = indexer[
            period + 1, educ_level, 1, exp_p + 1, exp_f, type_, 0, 0,  # No partner
        ]

        k_1_11 = indexer[
            period + 1, educ_level, 1, exp_p + 1, exp_f, type_, 0, 1,  # Partner
        ]

        # Choice: Full-time
        k_2_10 = indexer[
            period + 1, educ_level, 2, exp_p, exp_f + 1, type_, 0, 0,  # No partner
        ]

        k_2_11 = indexer[
            period + 1, educ_level, 2, exp_p, exp_f + 1, type_, 0, 1,  # Partner
        ]

        emaxs[k_parent, 0] = weight_emax_with_child(
            emax_01=emaxs[k_0_01, 3],
            emax_00=emaxs[k_0_00, 3],
            emax_10=emaxs[k_0_10, 3],
            emax_11=emaxs[k_0_11, 3],
            prob_kid=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )

        emaxs[k_parent, 1] = weight_emax_with_child(
            emax_01=emaxs[k_1_01, 3],
            emax_00=emaxs[k_1_00, 3],
            emax_10=emaxs[k_1_10, 3],
            emax_11=emaxs[k_1_11, 3],
            prob_kid=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )
        emaxs[k_parent, 2] = weight_emax_with_child(
            emax_01=emaxs[k_2_01, 3],
            emax_00=emaxs[k_2_00, 3],
            emax_10=emaxs[k_2_10, 3],
            emax_11=emaxs[k_2_11, 3],
            prob_kid=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )

    return emaxs


@numba.njit(nogil=True)
def weight_emax_with_child(emax_01, emax_00, emax_10, emax_11, prob_kid, prob_partner):
    weight_01 = (1 - prob_kid) * prob_partner[1] * emax_01
    weight_00 = (1 - prob_kid) * prob_partner[0] * emax_00
    weight_10 = prob_kid * prob_partner[0] * emax_10
    weight_11 = prob_kid * prob_partner[1] * emax_11
    return weight_11 + weight_10 + weight_00 + weight_01


# def get_child_states(
#         period,
#         educ_level,
#         child_lagged_choice,
#         child_exp_p,
#         child_exp_f,
#         type_,
#         child_age_update_rule):
#     emax_01 =
