import jax
import jax.numpy as jnp

from soepy.shared.tax_and_transfers_jax import calculate_net_income


def _get_max_aggregated_utilities(
    delta,
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


def construct_emax(
    delta,
    log_wages_systematic,
    non_consumption_utilities,
    draws,
    draw_weights,
    continuation_values,
    hours,
    mu,
    non_employment_consumption_resources,
    covariates,
    model_spec,
    tax_splitting,
):
    """Compute EMAX given already-weighted continuation values.

    Parameters
    ----------
    continuation_values : jax.numpy.ndarray, shape (n_states, 3)
        Expected continuation values for each choice.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_states, 4): continuation values + expected max value.
    """

    def per_draw(
        log_wage_state,
        non_cons_util_state,
        cont_state,
        non_emp_state,
        covariate,
        draw,
        draw_weight,
    ):
        male_wage = covariate[1]
        equivalence_scale = covariate[2]
        child_benefit = covariate[3]

        child_care_bin = jnp.where(covariate[0] > 2, 0, covariate[0]).astype(int)

        return _get_max_aggregated_utilities(
            delta=delta,
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
        )

    def per_state(
        log_wage_state,
        non_cons_util_state,
        cont_state,
        non_emp_state,
        covariate,
    ):
        weighted = jax.vmap(
            lambda draw, w: per_draw(
                log_wage_state,
                non_cons_util_state,
                cont_state,
                non_emp_state,
                covariate,
                draw,
                w,
            )
        )(draws, draw_weights)
        return weighted.sum(axis=0)

    emax = jax.vmap(per_state)(
        log_wages_systematic,
        non_consumption_utilities,
        continuation_values,
        non_employment_consumption_resources,
        covariates,
    )
    return jnp.concatenate([continuation_values, emax[:, jnp.newaxis]], axis=1)
