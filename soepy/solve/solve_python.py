import jax
import jax.numpy as jnp
import numpy as np

from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import get_integration_draws_and_weights
from soepy.shared.shared_auxiliary import calculate_log_wage
from soepy.shared.shared_auxiliary import calculate_non_consumption_utility
from soepy.shared.shared_constants import HOURS
from soepy.solve.emaxs import construct_emax
from soepy.solve.validation_solve import construct_emax_validation


def pyth_solve(
    states,
    covariates,
    child_state_indexes,
    model_params,
    model_spec,
    prob_child,
    prob_partner,
    is_expected,
):
    """Solve the model by backward induction.

    The solution routine performs four key operations:
    - create all nodes (state space points) of the decision tree (state space)
    that the agents might possibly reach.
    - create covariates that depend on the state space components at every
    state space point.
    - calculate the instantaneous/flow utilities for each possible choice at every
    state space point
    - calculate the continuation values for each choice at every
    state space point.

    Parameters
    __________
    model_params : namedtuple
        Namedtuple containing all structural, potentially free and estimable,
        parameters relevant for running a simulation.
    model_spec : namedtuple
        Namedtuple containing all fixed parameters relevant for running a simulation
    is_expected: bool
        A boolean indicator that differentiates between the human capital accumulation
        process that agents expect (is_expected = True) and that the market generates
        (is_expected = False)

    Returns
    _______
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.
    covariates : np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.
    emaxs : np.ndarray
        Array with shape (num states, num_choices +1). First block of dimension
        num_choices contains continuation values of the state space point.
        Lat element contains the expected maximum value function of the state space point.
    """

    draws_emax, draw_weights_emax = get_integration_draws_and_weights(
        model_spec, model_params
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs, non_consumption_utilities = pyth_backward_induction(
        model_spec,
        model_spec.tax_splitting,
        model_params,
        states,
        child_state_indexes,
        draws_emax,
        draw_weights_emax,
        covariates,
        prob_child,
        prob_partner,
        is_expected,
    )

    # Return function output
    return (
        non_consumption_utilities,
        emaxs,
    )


def pyth_backward_induction(
    model_spec,
    tax_splitting,
    model_params,
    states,
    child_state_indexes,
    draws,
    draw_weights,
    covariates,
    prob_child,
    prob_partner,
    is_expected,
):
    """Get expected maximum value function at every state space point.
    Backward induction is performed all at once for all states in a given period.
    The function loops through each period. The included construct_emax function
    implicitly loops through all states in the period currently reached by the
    parent loop.

    Parameters
    ----------
    model_spec : namedtuple
        Contains all fixed parameters of the model including information on dimensions
        such as number of periods, agents, random draws, etc.
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        Array where each dimension represents a componenet of the state space.
        :data:`states[k]` returns the values of the state space components
        at state :data:`k`. Indexing :data:`indexer` by the same state space
        component values returns :data:`k`.
    log_wage_systematic : np.array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.

    Returns
    -------
    emaxs : np.ndarray
        An array of dimension (num_states, num choices + 1). The object's rows contain
        the continuation values of each choice at the specific state space points
        as its first elements. The last row element corresponds to the maximum
        expected value function of the state.
    """

    hours = np.array(HOURS)
    non_consumption_utilities = calculate_non_consumption_utility(
        model_params,
        states,
        covariates[:, 0],
    )

    emaxs = np.zeros((states.shape[0], non_consumption_utilities.shape[1] + 1))

    partial_body = jax.jit(
        lambda params, period_index, emaxs_childs, prob_child_period, prob_partner_period: period_body_backward_induction(
            model_params=params,
            state_period_index=period_index,
            emaxs_child_states=emaxs_childs,
            states=jnp.asarray(states),
            covariates=jnp.asarray(covariates),
            non_consumption_utilities=jnp.asarray(non_consumption_utilities),
            prob_child_period=prob_child_period,
            prob_partner_period=prob_partner_period,
            draws=jnp.asarray(draws),
            draw_weights=jnp.asarray(draw_weights),
            model_spec=model_spec,
            is_expected=is_expected,
            hours=jnp.asarray(hours),
            tax_splitting=tax_splitting,
        )
    )

    # Loop backwards over all periods
    for period in np.arange(model_spec.num_periods - 1, -1, -1, dtype=int):
        bool_ind = states[:, 0] == period
        state_period_index = np.where(bool_ind)[0]
        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == model_spec.num_periods - 1:
            emaxs_child_states = jnp.zeros(
                shape=(state_period_index.shape[0], 3, 2, 2), dtype=float
            )
        else:
            child_states_ind_period = child_state_indexes[state_period_index]
            emaxs_child_states = emaxs[:, 3][child_states_ind_period]

        emaxs_period = partial_body(
            params=model_params,
            period_index=state_period_index,
            emaxs_childs=emaxs_child_states,
            prob_child_period=prob_child[period],
            prob_partner_period=prob_partner[period],
        )

        emaxs[state_period_index] = emaxs_period

    return emaxs, non_consumption_utilities


def period_body_backward_induction(
    model_params,
    state_period_index,
    emaxs_child_states,
    states,
    covariates,
    non_consumption_utilities,
    prob_child_period,
    prob_partner_period,
    draws,
    draw_weights,
    model_spec,
    is_expected,
    hours,
    tax_splitting,
):

    deductions_spec = model_spec.ssc_deductions
    tax_params = model_spec.tax_params
    child_care_costs = model_spec.child_care_costs

    erziehungsgeld_inc_single = model_spec.erziehungsgeld_income_threshold_single
    erziehungsgeld_inc_married = model_spec.erziehungsgeld_income_threshold_married
    erziehungsgeld = model_spec.erziehungsgeld

    # Extract period information
    # States and covariates
    states_period = states[state_period_index]
    covariates_period = covariates[state_period_index]
    non_consumption_utilities_period = non_consumption_utilities[state_period_index]

    # Corresponding equivalence scale for period states
    male_wage_period = covariates_period[:, 1]
    equivalence_scale_period = covariates_period[:, 2]
    child_benefits_period = covariates_period[:, 3]

    index_child_care_costs_period = jnp.where(
        covariates_period[:, 0] > 2, 0, covariates_period[:, 0]
    ).astype(int)

    # Probability that a child arrives
    prob_child_period_states = prob_child_period[states_period[:, 1]]

    # Probability of partner states.
    prob_partner_period_states = prob_partner_period[
        states_period[:, 1], states_period[:, 7]
    ]

    # Period rewards
    log_wage_systematic_period = calculate_log_wage(
        model_params, states_period, is_expected
    ) + np.log(model_spec.elasticity_scale)

    non_employment_consumption_resources_period = (
        calculate_non_employment_consumption_resources(
            deductions_spec=model_spec.ssc_deductions,
            income_tax_spec=model_spec.tax_params,
            model_spec=model_spec,
            states=states_period,
            log_wage_systematic=log_wage_systematic_period,
            male_wage=male_wage_period,
            child_benefits=child_benefits_period,
            tax_splitting=model_spec.tax_splitting,
            hours=hours,
        )
    )

    if model_spec.parental_leave_regime == "elterngeld":
        # Calculate emax for current period reached by the loop
        emaxs_period = construct_emax(
            model_params.delta,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            draws,
            draw_weights,
            emaxs_child_states,
            prob_child_period_states,
            prob_partner_period_states,
            hours,
            model_params.mu,
            non_employment_consumption_resources_period,
            deductions_spec,
            tax_params,
            child_care_costs,
            index_child_care_costs_period,
            male_wage_period,
            child_benefits_period,
            equivalence_scale_period,
            tax_splitting,
        )
    elif model_spec.parental_leave_regime == "erziehungsgeld":

        baby_child_period = (states_period[:, 6] == 0) | (states_period[:, 6] == 1)
        # Calculate emax for current period reached by the loop
        emaxs_period = construct_emax_validation(
            model_params.delta,
            baby_child_period,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            draws,
            draw_weights,
            emaxs_child_states,
            prob_child_period_states,
            prob_partner_period_states,
            hours,
            model_params.mu,
            non_employment_consumption_resources_period,
            deductions_spec,
            tax_params,
            child_care_costs,
            index_child_care_costs_period,
            male_wage_period,
            child_benefits_period,
            equivalence_scale_period,
            erziehungsgeld_inc_single,
            erziehungsgeld_inc_married,
            erziehungsgeld,
            tax_splitting,
        )

    else:
        raise ValueError(
            f"Parental leave regime {model_spec.parental_leave_regime} not specified."
        )

    return emaxs_period
