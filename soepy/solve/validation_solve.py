import jax
import jax.numpy as jnp

from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.tax_and_transfers_jax import calculate_net_income


def _get_max_aggregated_utilities_validation(
    *,
    delta,
    baby_child,
    log_wage_systematic,
    non_consumption_utilities,
    draw,
    draw_weight,
    emaxs,
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
    """
    JAX version of _get_max_aggregated_utilities_validation.
    Returns the weighted max utility for one draw.
    """

    # Choice 0: non-employment
    consumption_0 = non_employment_consumption_resources / equivalence
    current_max_value_function = (consumption_0**mu / mu) * non_consumption_utilities[
        0
    ] + delta * emaxs[0]

    # Choices 1,2: work (part-time, full-time)
    for j in range(1, 3):
        female_wage = hours[j] * jnp.exp(log_wage_systematic + draw)

        net_income = calculate_net_income(
            income_tax_spec, deductions_spec, female_wage, male_wage, tax_splitting
        )

        # Erziehungsgeld add-on: only if child benefits > 0 and j == 1 (part-time)
        # Original condition:
        #   (child_benefits > 0) * (j == 1) * calc_erziehungsgeld(...)
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
            consumption_utility * non_consumption_utilities[j] + delta * emaxs[j]
        )
        current_max_value_function = jnp.maximum(
            current_max_value_function, value_function_choice
        )

    return current_max_value_function * draw_weight


def do_weighting_emax_scalar(child_emaxs, prob_child, prob_partner):
    weight_01 = (1.0 - prob_child) * prob_partner[1] * child_emaxs[0, 1]
    weight_00 = (1.0 - prob_child) * prob_partner[0] * child_emaxs[0, 0]
    weight_10 = prob_child * prob_partner[0] * child_emaxs[1, 0]
    weight_11 = prob_child * prob_partner[1] * child_emaxs[1, 1]
    return weight_11 + weight_10 + weight_00 + weight_01


def emax_weighting(emaxs_child_states, prob_child, prob_partner):
    # Matches your style from the previous rewrite.
    return jax.vmap(
        jax.vmap(do_weighting_emax_scalar, in_axes=(0, None, None)),
        in_axes=(0, 0, 0),
    )(emaxs_child_states, prob_child, prob_partner)


def construct_emax_validation(
    delta,
    baby_child,
    log_wages_systematic,
    non_consumption_utilities,
    draws,
    draw_weights,
    emaxs_child_states,
    prob_child,
    prob_partner,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    child_care_costs,
    index_child_care_costs,
    male_wages,
    child_benefits,
    equivalence_scales,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
    tax_splitting,
):
    """Simulate expected maximum utility for a given distribution of the unobservables. The function calculates the
    maximum expected value function over the distribution of the error term at each state space point in the period
    currently reached by the parent loop. The expectation calculation is performed via Monte Carlo integration. The
    goal is to approximate an integral by evaluating the integrand at randomly chosen points. In this setting, one
    wants to approximate the expected maximum utility of a given state.

    """
    # Weighted continuation values for choices 0..2
    emax = emax_weighting(
        emaxs_child_states, prob_child, prob_partner
    )  # (num_states, 3)

    child_care_costs_j = jnp.asarray(child_care_costs)

    def max_aggregated_utilities_broadcast(
        log_wage_systematic_choices,
        non_consumption_utilities_choices,
        emax_choices,
        non_employment_consumption_resources_choice,
        male_wage,
        child_benefit,
        equivalence,
        index_child_care_cost,
        baby_child_scalar,
        draw,
        draw_weight,
    ):
        return _get_max_aggregated_utilities_validation(
            delta=delta,
            baby_child=baby_child_scalar,
            log_wage_systematic=log_wage_systematic_choices,
            non_consumption_utilities=non_consumption_utilities_choices,
            draw=draw,
            draw_weight=draw_weight,
            emaxs=emax_choices,
            hours=hours,
            mu=mu,
            non_employment_consumption_resources=non_employment_consumption_resources_choice,
            deductions_spec=deductions_spec,
            income_tax_spec=income_tax_spec,
            male_wage=male_wage,
            child_benefits=child_benefit,
            equivalence=equivalence,
            tax_splitting=tax_splitting,
            child_care_costs=child_care_costs_j,
            child_care_bin=index_child_care_cost,
            erziehungsgeld_inc_single=erziehungsgeld_inc_single,
            erziehungsgeld_inc_married=erziehungsgeld_inc_married,
            erziehungsgeld=erziehungsgeld,
        )

    # Shape expectations matching your previous rewrite:
    # Outer vmap over states; inner vmap over draws (draw and draw_weight are axis 0 in inner vmap).
    emaxs_current_states = jax.vmap(
        jax.vmap(
            max_aggregated_utilities_broadcast,
            in_axes=(None, None, None, None, None, None, None, None, None, 0, 0),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
    )(
        log_wages_systematic,
        non_consumption_utilities,
        emax,
        non_employment_consumption_resources,
        male_wages,
        child_benefits,
        equivalence_scales,
        index_child_care_costs,
        baby_child,
        draws,
        draw_weights,
    )

    # Sum over draws axis
    emax_expected = emaxs_current_states.sum(axis=1)

    # Return (num_states, 4): first 3 are continuation values; last is expected max utility
    emaxs_with_expected = jnp.concatenate([emax, emax_expected[:, None]], axis=1)
    return emaxs_with_expected
