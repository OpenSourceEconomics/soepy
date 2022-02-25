from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.config import config

from soepy.shared.non_employment_benefits import calculate_non_employment_benefits
from soepy.shared.shared_auxiliary import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_constants import HOURS
from soepy.shared.shared_constants import NUM_CHOICES
from soepy.solve.emaxs import vmap_construct_emax_jax

config.update("jax_enable_x64", True)


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

    attrs_spec = ["seed_emax", "num_periods", "num_draws_emax"]
    draws_emax = draw_disturbances(
        *[getattr(model_spec, attr) for attr in attrs_spec], model_params
    )

    log_wage_systematic, non_consumption_utilities = calculate_utility_components(
        model_params, model_spec, states, covariates, is_expected
    )

    non_employment_benefits = calculate_non_employment_benefits(
        model_spec, states, log_wage_systematic
    )

    non_employment_consumption_resources = vmap(
        partial(
            calculate_non_employment_consumption_resources,
            jnp.array(model_spec.ssc_deductions),
            jnp.array(model_spec.tax_params),
            model_spec.tax_splitting,
        ),
        in_axes=(0, 0, 0),
    )(
        jnp.array(covariates[:, 1]),
        jnp.array(non_employment_benefits),
        jnp.array(states[:, 7]),
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs = pyth_backward_induction(
        model_spec,
        states,
        child_state_indexes,
        log_wage_systematic,
        non_consumption_utilities,
        draws_emax,
        covariates,
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


def pyth_backward_induction(
    model_spec,
    states,
    child_state_indexes,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    covariates,
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
    emaxs = np.zeros((states.shape[0], NUM_CHOICES + 1))

    # Read relevant values from dictionary
    tax_splitting = model_spec.tax_splitting
    delta = model_spec.delta
    tax_params_jax = jnp.array(model_spec.tax_params)
    child_care_costs = jnp.array(model_spec.child_care_costs)
    mu = model_spec.mu
    hours = jnp.array(HOURS)
    deductions_spec_jax = jnp.array(deductions_spec)

    # Loop backwards over all periods
    for period in reversed(range(model_spec.num_periods)):
        state_period_index = np.where(states[:, 0] == period)[0]

        # Extract period information
        # States
        states_period = states[state_period_index]

        # Probability that a child arrives
        prob_child_period = jnp.take(prob_child[period], states_period[:, 1])

        # Probability of partner states.
        prob_partner_period_np = prob_partner[period][
            states_period[:, 1], states_period[:, 7]
        ]
        prob_partner_period = jnp.array(prob_partner_period_np)

        # Period rewards
        log_wage_systematic_period = jnp.take(log_wage_systematic, state_period_index)
        non_consumption_utilities_period = jnp.take(
            non_consumption_utilities, state_period_index, axis=0
        )
        non_employment_consumption_resources_period = jnp.take(
            non_employment_consumption_resources, state_period_index
        )

        # Corresponding equivalence scale for period states
        male_wage_period = jnp.take(covariates[:, 1], state_period_index)
        equivalence_scale_period = jnp.take(covariates[:, 2], state_period_index)
        child_benefits_period = jnp.take(covariates[:, 3], state_period_index)
        child_bins_period = jnp.take(covariates[:, 0].astype(int), state_period_index)
        index_child_care_costs = jnp.where(child_bins_period > 2, 0, child_bins_period)

        partner_indicator = jnp.array(states_period[:, 7])

        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == model_spec.num_periods - 1:
            emaxs_child_states = jnp.zeros(
                shape=(states_period.shape[0], 3, 2, 2), dtype=float
            )
        else:
            child_states_ind_period = jnp.take(
                child_state_indexes, state_period_index, axis=0
            )
            emaxs_child_states = jnp.take(emaxs[:, 3], child_states_ind_period, axis=0)

        # Calculate emax for current period reached by the loop
        emaxs_period = vmap_construct_emax_jax(
            delta,
            mu,
            tax_splitting,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            jnp.array(draws[period]),
            emaxs_child_states,
            prob_child_period,
            prob_partner_period,
            hours,
            non_employment_consumption_resources_period,
            deductions_spec_jax,
            tax_params_jax,
            child_care_costs,
            index_child_care_costs,
            male_wage_period,
            child_benefits_period,
            equivalence_scale_period,
            partner_indicator,
        )

        emaxs[state_period_index, :] = emaxs_period

    return emaxs
