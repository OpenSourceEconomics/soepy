import numba

from soepy.solve.create_state_space import get_child_states_index


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

        emaxs[k_parent, 0] = weight_emax(
            cont_emaxs=emaxs[:, 3],
            indexer=indexer,
            next_period=period + 1,
            educ_level=educ_level,
            child_lagged_choice=0,
            child_exp_p=exp_p,
            child_exp_f=exp_f,
            disutil_type=type_,
            child_age_update_rule_current_state=child_age_update_rule[k_parent],
            prob_child=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )

        emaxs[k_parent, 1] = weight_emax(
            cont_emaxs=emaxs[:, 3],
            indexer=indexer,
            next_period=period + 1,
            educ_level=educ_level,
            child_lagged_choice=1,
            child_exp_p=exp_p + 1,
            child_exp_f=exp_f,
            disutil_type=type_,
            child_age_update_rule_current_state=child_age_update_rule[k_parent],
            prob_child=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )
        emaxs[k_parent, 2] = weight_emax(
            cont_emaxs=emaxs[:, 3],
            indexer=indexer,
            next_period=period + 1,
            educ_level=educ_level,
            child_lagged_choice=2,
            child_exp_p=exp_p,
            child_exp_f=exp_f + 1,
            disutil_type=type_,
            child_age_update_rule_current_state=child_age_update_rule[k_parent],
            prob_child=prob_child_period[educ_level],
            prob_partner=prob_partner_process[educ_level, partner_indicator, :],
        )

    return emaxs


@numba.njit(nogil=True)
def weight_emax(
    cont_emaxs,
    indexer,
    next_period,
    educ_level,
    child_lagged_choice,
    child_exp_p,
    child_exp_f,
    disutil_type,
    child_age_update_rule_current_state,
    prob_child,
    prob_partner,
):
    child_indexes = get_child_states_index(
        indexer,
        next_period,
        educ_level,
        child_lagged_choice,
        child_exp_p,
        child_exp_f,
        disutil_type,
        child_age_update_rule_current_state,
    )
    return do_weighting_emax(cont_emaxs, child_indexes, prob_child, prob_partner)


@numba.njit(nogil=True)
def do_weighting_emax(cont_emaxs, child_indexes, prob_child, prob_partner):
    emax_01 = cont_emaxs[child_indexes[0, 1]]
    emax_00 = cont_emaxs[child_indexes[0, 0]]
    emax_11 = cont_emaxs[child_indexes[1, 1]]
    emax_10 = cont_emaxs[child_indexes[1, 0]]

    weight_01 = (1 - prob_child) * prob_partner[1] * emax_01
    weight_00 = (1 - prob_child) * prob_partner[0] * emax_00
    weight_10 = prob_child * prob_partner[0] * emax_10
    weight_11 = prob_child * prob_partner[1] * emax_11
    return weight_11 + weight_10 + weight_00 + weight_01
