import numpy as np
from scipy.special import roots_hermite

from soepy.shared.non_employment_benefits import calculate_non_employment_benefits
from soepy.shared.shared_auxiliary import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_constants import HOURS
from soepy.solve.emaxs import construct_emax


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
    log_wage_systematic, non_consumption_utilities = calculate_utility_components(
        model_params, model_spec, states, covariates, is_expected
    )

    non_employment_benefits = calculate_non_employment_benefits(
        model_spec, states, log_wage_systematic
    )

    tax_splitting = model_spec.tax_splitting

    non_employment_consumption_resources = (
        calculate_non_employment_consumption_resources(
            model_spec.ssc_deductions,
            model_spec.tax_params,
            covariates[:, 1],
            non_employment_benefits,
            tax_splitting,
        )
    )

    index_child_care_costs = np.where(covariates[:, 0] > 2, 0, covariates[:, 0]).astype(
        int
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs = pyth_backward_induction(
        model_spec.num_periods,
        tax_splitting,
        model_params.mu,
        model_params.delta,
        model_spec.tax_params,
        states,
        HOURS,
        model_spec.child_care_costs,
        child_state_indexes,
        log_wage_systematic,
        non_consumption_utilities,
        draws_emax,
        draw_weights_emax,
        covariates,
        index_child_care_costs,
        prob_child,
        prob_partner,
        non_employment_consumption_resources,
        model_spec.ssc_deductions,
    )

    # Return function output
    return (
        non_employment_consumption_resources,
        emaxs,
    )


def get_integration_draws_and_weights(model_spec, model_params):
    if model_spec.integration_method == "quadrature":
        # Draw standard points and corresponding weights
        standard_draws, draw_weights_emax = roots_hermite(model_spec.num_draws_emax)
        # Rescale draws and weights
        draws_emax = standard_draws * np.sqrt(2) * model_params.shock_sd
        draw_weights_emax *= 1 / np.sqrt(np.pi)
    elif model_spec.integration_method == "monte_carlo":
        draws_emax = draw_disturbances(
            model_spec.seed_emax, 1, model_spec.num_draws_emax, model_params
        )[0]
        draw_weights_emax = (
            np.ones(model_spec.num_draws_emax) / model_spec.num_draws_emax
        )
    else:
        raise ValueError(
            f"Integration method {model_spec.integration_method} not specified."
        )

    return draws_emax, draw_weights_emax


# @numba.njit
def pyth_backward_induction(
    num_periods,
    tax_splitting,
    mu,
    delta,
    tax_params,
    states,
    hours,
    child_care_costs,
    child_state_indexes,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    draw_weights,
    covariates,
    index_child_care_costs,
    prob_child,
    prob_partner,
    non_employment_consumption_resources,
    deductions_spec,
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
    dummy_array = np.zeros(4)  # Need this array to define output for construct_emaxs

    emaxs = np.zeros((states.shape[0], non_consumption_utilities.shape[1] + 1))

    # Loop backwards over all periods
    for period in np.arange(num_periods - 1, -1, -1, dtype=int):
        state_period_index = np.where(states[:, 0] == period)[0]

        # Extract period information
        # States
        states_period = states[state_period_index]

        # Probability that a child arrives
        prob_child_period = prob_child[period][states_period[:, 1]]

        # Probability of partner states.
        prob_partner_period = prob_partner[period][
            states_period[:, 1], states_period[:, 7]
        ]

        # Period rewards
        log_wage_systematic_period = log_wage_systematic[state_period_index]
        non_consumption_utilities_period = non_consumption_utilities[state_period_index]
        non_employment_consumption_resources_period = (
            non_employment_consumption_resources[state_period_index]
        )

        # Corresponding equivalence scale for period states
        covariates_state = covariates[state_period_index]
        male_wage_period = covariates_state[:, 1]
        equivalence_scale_period = covariates_state[:, 2]
        child_benefits_period = covariates_state[:, 3]

        index_child_care_costs_period = index_child_care_costs[state_period_index]

        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == num_periods - 1:
            emaxs_child_states = np.zeros(
                shape=(states_period.shape[0], 3, 2, 2), dtype=float
            )
        else:
            child_states_ind_period = child_state_indexes[state_period_index]
            emaxs_child_states = emaxs[:, 3][child_states_ind_period]

        # Calculate emax for current period reached by the loop
        emaxs_period = construct_emax(
            delta,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            draws,
            draw_weights,
            emaxs_child_states,
            prob_child_period,
            prob_partner_period,
            hours,
            mu,
            non_employment_consumption_resources_period,
            deductions_spec,
            tax_params,
            child_care_costs,
            index_child_care_costs_period,
            male_wage_period,
            child_benefits_period,
            equivalence_scale_period,
            tax_splitting,
            dummy_array,
        )

        emaxs[state_period_index] = emaxs_period

    return emaxs
