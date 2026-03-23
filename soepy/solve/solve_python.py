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
from soepy.shared.non_consumption_utility import calculate_non_consumption_utility
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import get_integration_draws_and_weights
from soepy.shared.wages import calculate_log_wage
from soepy.solve.continuous_continuation import (
    interpolate_then_weight_continuation_values,
)
from soepy.solve.emaxs import construct_emax
from soepy.solve.terminal_proxy import terminal_proxy_continuation
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

    # Leave large arrys as inputs when generating the function to be jitted.
    func_to_jit = lambda params_arg, states_arg, covariates_arg, child_state_indexes_local_arg, unscaled_draws_emax_arg, draw_weights_emax_arg, prob_child_arg, prob_partner_arg: pyth_backward_induction(
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

    solve_function = lambda params: jax.jit(func_to_jit)(
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

    states_last = states_per_period[-1]
    covariates_last = covariates_per_period[-1]
    current_period_last = states_last[0, PERIOD]

    terminal_continuation = terminal_proxy_continuation(
        exp_grid=exp_grid,
        states_period=states_last,
        covariates_period=covariates_last,
        model_params=model_params,
        model_spec=model_spec,
        biased_exp=biased_exp,
        current_period=current_period_last,
    )

    emaxs_last, non_consumption_utilities_last = solve_period_emax(
        states_period=states_last,
        covariates_period=covariates_last,
        continuation_values=terminal_continuation,
        exp_grid=exp_grid,
        model_params=model_params,
        model_spec=model_spec,
        hours=hours,
        draws=draws,
        draw_weights=draw_weights,
        current_period=current_period_last,
    )

    period_specific_objects_without_last = jax.tree_util.tree_map(
        lambda a: a[1:], period_specific_objects_rev
    )

    scan_step = lambda emaxs_next, period_data: scan_step_with_interpolation(
        emaxs_next=emaxs_next,
        period_data=period_data,
        exp_grid=exp_grid,
        model_params=model_params,
        model_spec=model_spec,
        hours=hours,
        biased_exp=biased_exp,
        draws=draws,
        draw_weights=draw_weights,
    )

    _, (emaxs_rev, non_consumption_utilities_rev) = jax.lax.scan(
        scan_step, emaxs_last, period_specific_objects_without_last
    )

    non_consumption_utilities, emaxs = stack_backward_outputs(
        emaxs_rev=emaxs_rev,
        non_consumption_utilities_rev=non_consumption_utilities_rev,
        emaxs_last=emaxs_last,
        non_consumption_utilities_last=non_consumption_utilities_last,
    )

    return non_consumption_utilities, emaxs


def scan_step_with_interpolation(
    emaxs_next,
    period_data,
    exp_grid,
    model_params,
    model_spec,
    hours,
    biased_exp,
    draws,
    draw_weights,
):
    states_period = period_data["states"]
    covariates_period = period_data["covariates"]
    child_state_indexes_local = period_data["child_state_indexes_local"]
    prob_child_period = period_data["prob_child"]
    prob_partner_period = period_data["prob_partner"]

    current_period = states_period[0, PERIOD]
    edu_state = states_period[:, EDUC_LEVEL]
    partner_state = states_period[:, PARTNER]

    prob_child_period_states = prob_child_period[edu_state]
    prob_partner_period_states = prob_partner_period[edu_state, partner_state]

    continuation_values = compute_continuation_values(
        exp_grid=exp_grid,
        emaxs_next=emaxs_next,
        child_state_indexes_local=child_state_indexes_local,
        current_period=current_period,
        model_params=model_params,
        model_spec=model_spec,
        educ_level=edu_state,
        child_age=states_period[:, AGE_YOUNGEST_CHILD],
        biased_exp=biased_exp,
        prob_child_states=prob_child_period_states,
        prob_partner_states=prob_partner_period_states,
    )

    emaxs_curr, non_consumption_utilities_period = solve_period_emax(
        states_period=states_period,
        covariates_period=covariates_period,
        continuation_values=continuation_values,
        exp_grid=exp_grid,
        model_params=model_params,
        model_spec=model_spec,
        hours=hours,
        draws=draws,
        draw_weights=draw_weights,
        current_period=current_period,
    )

    return emaxs_curr, (emaxs_curr, non_consumption_utilities_period)


def compute_continuation_values(
    exp_grid,
    emaxs_next,
    child_state_indexes_local,
    current_period,
    model_params,
    model_spec,
    educ_level,
    child_age,
    biased_exp,
    prob_child_states,
    prob_partner_states,
):
    return interpolate_then_weight_continuation_values(
        exp_grid=exp_grid,
        v_next_grid=emaxs_next[:, :, 3],
        child_state_indexes_local=child_state_indexes_local,
        period=current_period,
        init_exp_max=model_spec.init_exp_max,
        model_params=model_params,
        educ_level=educ_level,
        child_age=child_age,
        biased_exp=biased_exp,
        prob_child_states=prob_child_states,
        prob_partner_states=prob_partner_states,
    )


def solve_period_emax(
    states_period,
    covariates_period,
    continuation_values,
    exp_grid,
    model_params,
    model_spec,
    hours,
    draws,
    draw_weights,
    current_period,
):
    edu_state = states_period[:, EDUC_LEVEL]
    unobs_types = states_period[:, TYPE]

    log_wage_systematic_period = jax.vmap(
        lambda exp_stock: calculate_log_wage(
            model_params=model_params,
            educ=edu_state,
            period=current_period,
            init_exp_max=model_spec.init_exp_max,
            exp_stock=exp_stock,
        )
        + np.log(model_spec.elasticity_scale)
    )(exp_grid).T

    non_consumption_utilities_period = calculate_non_consumption_utility(
        model_params=model_params,
        educ=edu_state,
        unobs_type=unobs_types,
        child_bin=covariates_period[:, 0],
    )

    (
        non_employment_consumption_resources_period,
        _,
    ) = calculate_non_employment_consumption_resources(
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

    emax_fn = lambda log_wage_g, cont_g, non_emp_g: construct_emax(
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

    if model_spec.parental_leave_regime != "elterngeld":
        baby_child_period = (states_period[:, 4] == 0) | (states_period[:, 4] == 1)
        emax_fn = lambda log_wage_g, cont_g, non_emp_g: construct_emax_validation(
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

    emaxs_curr = jax.vmap(emax_fn, in_axes=(1, 2, 1), out_axes=1)(
        log_wage_systematic_period,
        continuation_values,
        non_employment_consumption_resources_period,
    )

    return emaxs_curr, non_consumption_utilities_period


def stack_backward_outputs(
    emaxs_rev,
    non_consumption_utilities_rev,
    emaxs_last,
    non_consumption_utilities_last,
):
    emaxs_head = jnp.flip(emaxs_rev, axis=0).reshape(
        -1, emaxs_rev.shape[2], emaxs_rev.shape[-1]
    )
    non_consumption_utilities_head = jnp.flip(
        non_consumption_utilities_rev, axis=0
    ).reshape(
        -1,
        non_consumption_utilities_rev.shape[2],
    )

    emaxs_tail = emaxs_last
    non_consumption_utilities_tail = non_consumption_utilities_last

    emaxs = jnp.concatenate([emaxs_head, emaxs_tail], axis=0)
    non_consumption_utilities = jnp.concatenate(
        [non_consumption_utilities_head, non_consumption_utilities_tail], axis=0
    )

    return non_consumption_utilities, emaxs
