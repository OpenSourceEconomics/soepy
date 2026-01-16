import jax
import jax.numpy as jnp
import numpy as np

from soepy.shared.non_consumption_utility import calculate_non_consumption_utility
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import get_integration_draws_and_weights
from soepy.shared.shared_constants import HOURS
from soepy.shared.shared_constants import NUM_CHOICES
from soepy.shared.wages import calculate_log_wage
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
    solve_func = get_solve_function(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        is_expected=is_expected,
    )
    non_consumption_utilities, emaxs = solve_func(model_params)

    # Return function output
    return (
        non_consumption_utilities,
        emaxs,
    )


def get_solve_function(
    states,
    covariates,
    child_state_indexes,
    model_spec,
    prob_child,
    prob_partner,
    is_expected,
):
    """Return the solve function used in the model."""
    # Draw integration draws and weights for EMAX calculation
    unscaled_draws_emax, draw_weights_emax = get_integration_draws_and_weights(
        model_spec
    )

    tax_splitting = model_spec.tax_splitting

    # Make all arrays in model params jax arrays
    # Transform model specs and model params to jax arrays
    model_spec = jax.tree_util.tree_map(lambda x: try_array(x), model_spec)

    hours = jnp.array(HOURS)

    n_periods = model_spec.num_periods
    n_states_per_period = int(states.shape[0] / n_periods)

    # Reshape into period-major blocks.
    states_pp = jnp.asarray(
        states.reshape(n_periods, n_states_per_period, states.shape[1])
    )
    covariates_pp = jnp.asarray(
        covariates.reshape(n_periods, n_states_per_period, covariates.shape[1])
    )

    child_state_indexes_pp = jnp.asarray(
        child_state_indexes.reshape(
            n_periods,
            n_states_per_period,
            child_state_indexes.shape[1],
            child_state_indexes.shape[2],
            child_state_indexes.shape[3],
        )
    )

    # Convert global child indices to *local indices of the next-period block*.
    # This keeps the scan step free of any state-space indexing logic.
    #
    # For period t, child states live in period t+1 whose block starts at (t+1)*n_states_per_period.
    child_state_indexes_local_pp = jnp.asarray(
        child_state_indexes_pp
        - (np.arange(n_periods)[:, None, None, None, None] + 1) * n_states_per_period
    )

    unscaled_draws_emax = jnp.asarray(unscaled_draws_emax)
    draw_weights_emax = jnp.asarray(draw_weights_emax)
    prob_child = jnp.asarray(prob_child)
    prob_partner = jnp.asarray(prob_partner)

    # Generate closure
    def func_to_jit(
        params_arg,
        states_arg,
        covariates_arg,
        child_state_indexes_local_arg,
        unscaled_draws_emax_arg,
        draw_weights_emax_arg,
        prob_child_arg,
        prob_partner_arg,
    ):
        return pyth_backward_induction(
            model_params=params_arg,
            states_per_period=states_arg,
            covariates_per_period=covariates_arg,
            child_state_indexes_local_per_period=child_state_indexes_local_arg,
            draws=unscaled_draws_emax_arg * params_arg.shock_sd,
            draw_weights=draw_weights_emax_arg,
            prob_child=prob_child_arg,
            prob_partner=prob_partner_arg,
            model_spec=model_spec,
            hours=hours,
            is_expected=is_expected,
            tax_splitting=tax_splitting,
        )

    # Create solve function to jit
    def solve_function(params):
        params_int = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)

        non_consumption_utilities, emaxs = jax.jit(func_to_jit)(
            params_arg=params_int,
            states_arg=states_pp,
            covariates_arg=covariates_pp,
            child_state_indexes_local_arg=child_state_indexes_local_pp,
            unscaled_draws_emax_arg=unscaled_draws_emax,
            draw_weights_emax_arg=draw_weights_emax,
            prob_child_arg=prob_child,
            prob_partner_arg=prob_partner,
        )
        return non_consumption_utilities, emaxs

    return solve_function


def pyth_backward_induction(
    model_params,
    states_per_period,
    covariates_per_period,
    child_state_indexes_local_per_period,
    draws,
    draw_weights,
    prob_child,
    prob_partner,
    model_spec,
    hours,
    is_expected,
    tax_splitting,
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
    # Convert inputs once to JAX arrays (scan body stays pure and compilation-friendly).
    period_specific_objects = {
        "states": states_per_period,
        "covariates": covariates_per_period,
        "child_state_indexes_local": child_state_indexes_local_per_period,
        "prob_child": prob_child,
        "prob_partner": prob_partner,
    }

    # Reverse time for backward induction (scan goes forward over reversed time).
    period_specific_objects_rev = jax.tree_util.tree_map(
        lambda a: a[::-1], period_specific_objects
    )

    # Initial "next-period" emaxs: terminal continuation values are zero.
    emaxs_next_init = jnp.zeros(
        (states_per_period.shape[1], NUM_CHOICES + 1), dtype=float
    )

    def scan_step(emaxs_next, period_data):
        """One backward-induction step over a single period block.

        Carry
        -----
        emaxs_next : array
            Emax array for the next period in time (already computed in the scan).

        period_data : dict (pytree)
            Period-specific arrays for states, covariates, transition indices, and
            probability objects.

        Returns
        -------
        carry : array
            The current period's emax array (becomes next carry).
        out : array
            The current period's emax array (collected over time by scan).
        """
        states_period = period_data["states"]
        covariates_period = period_data["covariates"]
        child_state_indexes_local = period_data["child_state_indexes_local"]
        prob_child_period = period_data["prob_child"]
        prob_partner_period = period_data["prob_partner"]

        # Continuation values are the maximum value function of child states.
        # The child maximum value function lives in the last column (index 3).
        emaxs_child_states = emaxs_next[:, 3][child_state_indexes_local]
        # ---------------------------------------------------------------------
        # Period reward and expectation computation.
        # ---------------------------------------------------------------------
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

        non_consumption_utilities_period = calculate_non_consumption_utility(
            model_params,
            states_period,
            covariates_period[:, 0],
        )

        non_employment_consumption_resources_period = (
            calculate_non_employment_consumption_resources(
                deductions_spec=model_spec.ssc_deductions,
                income_tax_spec=model_spec.tax_params,
                model_spec=model_spec,
                states=states_period,
                log_wage_systematic=log_wage_systematic_period,
                male_wage=covariates_period[:, 1],
                child_benefits=covariates_period[:, 3],
                tax_splitting=model_spec.tax_splitting,
                hours=hours,
            )
        )

        if model_spec.parental_leave_regime == "elterngeld":
            emaxs_curr = construct_emax(
                delta=model_params.delta,
                log_wages_systematic=log_wage_systematic_period,
                non_consumption_utilities=non_consumption_utilities_period,
                draws=draws,
                draw_weights=draw_weights,
                emaxs_child_states=emaxs_child_states,
                prob_child=prob_child_period_states,
                prob_partner=prob_partner_period_states,
                hours=hours,
                mu=model_params.mu,
                non_employment_consumption_resources=non_employment_consumption_resources_period,
                covariates=covariates_period,
                model_spec=model_spec,
                tax_splitting=tax_splitting,
            )
        elif model_spec.parental_leave_regime == "erziehungsgeld":
            baby_child_period = (states_period[:, 6] == 0) | (states_period[:, 6] == 1)

            emaxs_curr = construct_emax_validation(
                delta=model_params.delta,
                baby_child=baby_child_period,
                log_wages_systematic=log_wage_systematic_period,
                non_consumption_utilities=non_consumption_utilities_period,
                draws=draws,
                draw_weights=draw_weights,
                emaxs_child_states=emaxs_child_states,
                prob_child=prob_child_period_states,
                prob_partner=prob_partner_period_states,
                hours=hours,
                mu=model_params.mu,
                non_employment_consumption_resources=non_employment_consumption_resources_period,
                model_spec=model_spec,
                covariates=covariates_period,
                tax_splitting=tax_splitting,
            )
        else:
            raise ValueError(
                f"Parental leave regime {model_spec.parental_leave_regime} not specified."
            )

        # Current period becomes the next-period carry for the following (earlier) step.
        return emaxs_curr, (emaxs_curr, non_consumption_utilities_period)

    # Run backward induction: outputs are in reverse time order (terminal -> first).
    _, (emaxs_rev, non_consumption_utilities_rev) = jax.lax.scan(
        scan_step, emaxs_next_init, period_specific_objects_rev
    )

    # Flip back to chronological order and flatten to (num_states, NUM_CHOICES + 1).
    emaxs_flat = jnp.flip(emaxs_rev, axis=0).reshape(-1, emaxs_rev.shape[-1])
    non_consumption_utilities = jnp.flip(non_consumption_utilities_rev, axis=0).reshape(
        -1,
        non_consumption_utilities_rev.shape[2],
    )

    return non_consumption_utilities, emaxs_flat


def try_array(x):
    """Try to convert x to a jax array, otherwise return x unchanged."""
    try:
        return jnp.asarray(x)
    except Exception:
        return x
