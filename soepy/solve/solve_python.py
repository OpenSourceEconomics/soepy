import jax
import jax.numpy as jnp
import numpy as np

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.constants_and_indices import PARTNER
from soepy.shared.constants_and_indices import PERIOD
from soepy.shared.constants_and_indices import TYPE
from soepy.shared.experience_stock import get_pt_increment
from soepy.shared.non_consumption_utility import calculate_non_consumption_utility
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import get_integration_draws_and_weights
from soepy.shared.wages import calculate_log_wage
from soepy.solve.continuous_continuation import (
    interpolate_then_weight_continuation_values,
)
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
    biased_exp,
):
    solve_func = get_solve_function(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=biased_exp,
    )
    non_consumption_utilities, emaxs = solve_func(model_params)
    return non_consumption_utilities, emaxs


def get_solve_function(
    states,
    covariates,
    child_state_indexes,
    model_spec,
    prob_child,
    prob_partner,
    biased_exp,
):
    unscaled_draws_emax, draw_weights_emax = get_integration_draws_and_weights(
        model_spec
    )

    hours = jnp.array(HOURS)

    n_periods = model_spec.num_periods
    n_states_per_period = int(states.shape[0] / n_periods)

    states_pp = states.reshape(n_periods, n_states_per_period, states.shape[1])
    covariates_pp = covariates.reshape(
        n_periods, n_states_per_period, covariates.shape[1]
    )

    child_state_indexes_pp = child_state_indexes.reshape(
        n_periods,
        n_states_per_period,
        child_state_indexes.shape[1],
        child_state_indexes.shape[2],
        child_state_indexes.shape[3],
    )

    child_state_indexes_local_pp = (
        child_state_indexes_pp
        - (np.arange(n_periods)[:, None, None, None, None] + 1) * n_states_per_period
    )

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
            biased_exp=biased_exp,
        )

    def solve_function(params):
        return jax.jit(func_to_jit)(
            params_arg=params,
            states_arg=states_pp,
            covariates_arg=covariates_pp,
            child_state_indexes_local_arg=child_state_indexes_local_pp,
            unscaled_draws_emax_arg=unscaled_draws_emax,
            draw_weights_emax_arg=draw_weights_emax,
            prob_child_arg=prob_child,
            prob_partner_arg=prob_partner,
        )

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
    biased_exp,
):
    period_specific_objects = {
        "states": states_per_period,
        "covariates": covariates_per_period,
        "child_state_indexes_local": child_state_indexes_local_per_period,
        "prob_child": prob_child,
        "prob_partner": prob_partner,
    }

    period_specific_objects_rev = jax.tree_util.tree_map(
        lambda a: a[::-1], period_specific_objects
    )

    exp_grid = model_spec.exp_grid
    exp_grid_size = exp_grid.shape[0]

    emaxs_next_init = jnp.zeros(
        (states_per_period.shape[1], exp_grid_size, NUM_CHOICES + 1), dtype=float
    )

    def scan_step(emaxs_next, period_data):
        states_period = period_data["states"]
        covariates_period = period_data["covariates"]
        child_state_indexes_local = period_data["child_state_indexes_local"]
        prob_child_period = period_data["prob_child"]
        prob_partner_period = period_data["prob_partner"]

        current_period = states_period[0, PERIOD]
        edu_state = states_period[:, EDUC_LEVEL]
        partner_state = states_period[:, PARTNER]
        unobs_types = states_period[:, TYPE]

        pt_increment_states = get_pt_increment(
            model_params=model_params,
            educ_level=edu_state,
            child_age=states_period[:, AGE_YOUNGEST_CHILD],
            biased_exp=biased_exp,
        )

        prob_child_period_states = prob_child_period[edu_state]
        prob_partner_period_states = prob_partner_period[edu_state, partner_state]

        # Determine continuation values by interpolation on future grid and weighting by exogenous process
        # probabilities
        continuation_values = interpolate_then_weight_continuation_values(
            exp_grid=exp_grid,
            v_next_grid=emaxs_next[:, :, 3],
            child_state_indexes_local=child_state_indexes_local,
            period=current_period,
            init_exp_max=model_spec.init_exp_max,
            pt_increment_states=pt_increment_states,
            prob_child_states=prob_child_period_states,
            prob_partner_states=prob_partner_period_states,
        )
        # continuation_values: (n_states, NUM_CHOICES, n_grid)

        # Batch over wage calculation for the experience grid
        def log_wage_one_exp_grid(exp_stock):
            return calculate_log_wage(
                model_params=model_params,
                educ=edu_state,
                period=current_period,
                init_exp_max=model_spec.init_exp_max,
                pt_increment=pt_increment_states,
                exp_stock=exp_stock,
            ) + np.log(model_spec.elasticity_scale)

        log_wage_systematic_period = jax.vmap(log_wage_one_exp_grid)(exp_grid).T

        non_consumption_utilities_period = calculate_non_consumption_utility(
            model_params=model_params,
            educ=edu_state,
            unobs_type=unobs_types,
            child_bin=covariates_period[:, 0],
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

        def solve_one_gridpoint(log_wage_g, cont_g, non_emp_g):
            if model_spec.parental_leave_regime == "elterngeld":
                return construct_emax(
                    delta=model_params.delta,
                    log_wages_systematic=log_wage_g,
                    non_consumption_utilities=non_consumption_utilities_period,
                    draws=draws,
                    draw_weights=draw_weights,
                    continuation_values=cont_g,
                    hours=hours,
                    mu=model_params.mu,
                    non_employment_consumption_resources=non_emp_g,
                    covariates=covariates_period,
                    model_spec=model_spec,
                    tax_splitting=model_spec.tax_splitting,
                )

            else:
                baby_child_period = (states_period[:, 4] == 0) | (
                    states_period[:, 4] == 1
                )
                return construct_emax_validation(
                    delta=model_params.delta,
                    baby_child=baby_child_period,
                    log_wages_systematic=log_wage_g,
                    non_consumption_utilities=non_consumption_utilities_period,
                    draws=draws,
                    draw_weights=draw_weights,
                    continuation_values=cont_g,
                    hours=hours,
                    mu=model_params.mu,
                    non_employment_consumption_resources=non_emp_g,
                    model_spec=model_spec,
                    covariates=covariates_period,
                    tax_splitting=model_spec.tax_splitting,
                )

        emaxs_curr = jax.vmap(solve_one_gridpoint, in_axes=(1, 2, 1), out_axes=1,)(
            log_wage_systematic_period,
            continuation_values,
            non_employment_consumption_resources_period,
        )

        return emaxs_curr, (emaxs_curr, non_consumption_utilities_period)

    _, (emaxs_rev, non_consumption_utilities_rev) = jax.lax.scan(
        scan_step, emaxs_next_init, period_specific_objects_rev
    )

    emaxs = jnp.flip(emaxs_rev, axis=0).reshape(
        -1, emaxs_rev.shape[2], emaxs_rev.shape[-1]
    )
    non_consumption_utilities = jnp.flip(non_consumption_utilities_rev, axis=0).reshape(
        -1,
        non_consumption_utilities_rev.shape[2],
    )

    return non_consumption_utilities, emaxs
