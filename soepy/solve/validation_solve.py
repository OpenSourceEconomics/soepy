import jax
import jax.numpy as jnp

from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.tax_and_transfers_jax import calculate_net_income


def _get_max_aggregated_utilities(
    *,
    delta,
    baby_child,
    log_wage_systematic,
    non_consumption_utilities,
    draw,
    draw_weight,
    continuation_values,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    male_wage,
    child_benefits,
    equivalence,
    tax_splitting,
    child_care_costs,
    child_care_bin,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
):
    consumption_0 = non_employment_consumption_resources / equivalence
    current_max_value_function = (consumption_0**mu / mu) * non_consumption_utilities[
        0
    ] + delta * continuation_values[0]

    for j in range(1, 3):
        female_wage = hours[j] * jnp.exp(log_wage_systematic + draw)

        net_income = calculate_net_income(
            income_tax_spec,
            deductions_spec,
            female_wage,
            male_wage,
            tax_splitting,
        )

        eg = calc_erziehungsgeld(
            male_wage,
            female_wage,
            male_wage > 0,
            baby_child,
            erziehungsgeld_inc_single,
            erziehungsgeld_inc_married,
            erziehungsgeld,
        )
        net_income = net_income + jnp.where((child_benefits > 0) & (j == 1), eg, 0.0)

        child_costs = child_care_costs[child_care_bin, j - 1]

        consumption = (net_income + child_benefits - child_costs).clip(
            min=1e-14
        ) / equivalence
        consumption_utility = consumption**mu / mu

        value_function_choice = (
            consumption_utility * non_consumption_utilities[j]
            + delta * continuation_values[j]
        )
        current_max_value_function = jnp.maximum(
            current_max_value_function, value_function_choice
        )

    return current_max_value_function * draw_weight


def construct_emax_validation(
    delta,
    baby_child,
    log_wages_systematic,
    non_consumption_utilities,
    draws,
    draw_weights,
    continuation_values,
    hours,
    mu,
    non_employment_consumption_resources,
    model_spec,
    covariates,
    tax_splitting,
):
    """Validation-regime EMAX with pre-weighted continuation values."""

    def per_draw(
        log_wage_state,
        non_cons_util_state,
        cont_state,
        non_emp_state,
        covariate,
        baby_child_state,
        draw,
        draw_weight,
    ):
        male_wage = covariate[1]
        equivalence_scale = covariate[2]
        child_benefit = covariate[3]

        child_care_bin = jnp.where(covariate[0] > 2, 0, covariate[0]).astype(int)

        return _get_max_aggregated_utilities(
            delta=delta,
            baby_child=baby_child_state,
            log_wage_systematic=log_wage_state,
            non_consumption_utilities=non_cons_util_state,
            draw=draw,
            draw_weight=draw_weight,
            continuation_values=cont_state,
            hours=hours,
            mu=mu,
            non_employment_consumption_resources=non_emp_state,
            deductions_spec=model_spec.ssc_deductions,
            income_tax_spec=model_spec.tax_params,
            male_wage=male_wage,
            child_benefits=child_benefit,
            equivalence=equivalence_scale,
            tax_splitting=tax_splitting,
            child_care_costs=model_spec.child_care_costs,
            child_care_bin=child_care_bin,
            erziehungsgeld_inc_single=model_spec.erziehungsgeld_income_threshold_single,
            erziehungsgeld_inc_married=model_spec.erziehungsgeld_income_threshold_married,
            erziehungsgeld=model_spec.erziehungsgeld,
        )

    def per_state(
        log_wage_state,
        non_cons_util_state,
        cont_state,
        non_emp_state,
        covariate,
        baby_child_state,
    ):
        weighted = jax.vmap(
            lambda draw, w: per_draw(
                log_wage_state,
                non_cons_util_state,
                cont_state,
                non_emp_state,
                covariate,
                baby_child_state,
                draw,
                w,
            )
        )(draws, draw_weights)
        emax_expected = weighted.sum(axis=0)
        return jnp.concatenate([cont_state, emax_expected[None]], axis=0)

    return jax.vmap(per_state)(
        log_wages_systematic,
        non_consumption_utilities,
        continuation_values,
        non_employment_consumption_resources,
        covariates,
        baby_child,
    )
