import jax.numpy as jnp

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import LAGGED_CHOICE
from soepy.shared.constants_and_indices import PARTNER
from soepy.shared.tax_and_transfers_jax import calculate_net_income


def calculate_non_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    model_spec,
    states,
    log_wage_systematic,
    male_wage,
    child_benefits,
    tax_splitting,
    hours,
):
    non_employment_benefits = calculate_non_employment_benefits(
        hours=hours,
        states=states,
        log_wage_systematic=log_wage_systematic,
        child_benefit=child_benefits,
        male_wage=male_wage,
        model_spec=model_spec,
    )

    if non_employment_benefits.ndim == 2:
        male_wage = male_wage[:, None]
        female_wage = jnp.zeros_like(male_wage)
    else:
        female_wage = 0

    male_net_income = calculate_net_income(
        income_tax_spec=income_tax_spec,
        deductions_spec=deductions_spec,
        female_wage=female_wage,
        male_wage=male_wage,
        tax_splitting=tax_splitting,
    )

    return male_net_income + non_employment_benefits, male_net_income


def calculate_non_employment_benefits(
    hours,
    states,
    log_wage_systematic,
    child_benefit,
    male_wage,
    model_spec,
):
    no_child = states[:, AGE_YOUNGEST_CHILD] == -1
    working_ft_last_period = states[:, LAGGED_CHOICE] == 2
    working_pt_last_period = states[:, LAGGED_CHOICE] == 1
    married = states[:, PARTNER] == 1

    prox_net_wage_systematic = 0.65 * jnp.exp(log_wage_systematic)

    # We use this check to see if log_wage_systematic is batched (2D) or not (1D), i.e. if it is for each agent/state
    # or additionally for multiple experience grid points.
    if prox_net_wage_systematic.ndim == 2:
        # Broadcast state-dependent scalars to (n_states, 1).
        no_child = no_child[:, None]
        working_ft_last_period = working_ft_last_period[:, None]
        working_pt_last_period = working_pt_last_period[:, None]
        married = married[:, None]
        male_wage = male_wage[:, None]
        child_benefit = child_benefit[:, None]

    alg2_single = model_spec.regelsatz_single + model_spec.housing_single

    alg_2_alleinerziehend = (
        model_spec.regelsatz_single
        + model_spec.regelsatz_child
        + model_spec.addition_child_single
        + model_spec.housing_single
        + model_spec.housing_addtion
    )

    alg2 = calculate_alg2(
        no_child=no_child,
        married=married,
        alg2_single=alg2_single,
        alg_2_alleinerziehend=alg_2_alleinerziehend,
    )

    if model_spec.parental_leave_regime == "elterngeld":
        newborn_child = states[:, AGE_YOUNGEST_CHILD] == 0
        if prox_net_wage_systematic.ndim == 2:
            newborn_child = newborn_child[:, None]

        elterngeld = calculate_elterngeld(
            hours=hours,
            working_ft_last_period=working_ft_last_period,
            working_pt_last_period=working_pt_last_period,
            prox_net_wage_systematic=prox_net_wage_systematic,
            elterngeld_replacement=model_spec.elterngeld_replacement,
            elterngeld_min=model_spec.elterngeld_min,
            elterngeld_max=model_spec.elterngeld_max,
            child_benefit=child_benefit,
        )

        alg1 = calculate_alg1(
            hours=hours,
            working_ft_last_period=working_ft_last_period,
            working_pt_last_period=working_pt_last_period,
            no_child=no_child,
            prox_net_wage_systematic=prox_net_wage_systematic,
            alg1_replacement_no_child=model_spec.alg1_replacement_no_child,
            alg1_replacement_child=model_spec.alg1_replacement_child,
            child_benefit_if_child=child_benefit,
        )

        last_working_non_employment_benefits = (
            1 - newborn_child
        ) * alg1 + newborn_child * elterngeld
        non_employment_benefits = last_working_non_employment_benefits.clip(min=alg2)
    elif model_spec.parental_leave_regime == "erziehungsgeld":
        non_employment_benefits = calculate_alg1(
            hours=hours,
            working_ft_last_period=working_ft_last_period,
            working_pt_last_period=working_pt_last_period,
            no_child=no_child,
            prox_net_wage_systematic=prox_net_wage_systematic,
            alg1_replacement_no_child=model_spec.alg1_replacement_no_child,
            alg1_replacement_child=model_spec.alg1_replacement_child,
            child_benefit_if_child=child_benefit,
        ).clip(min=alg2)

        baby_child = (states[:, AGE_YOUNGEST_CHILD] == 0) | (
            states[:, AGE_YOUNGEST_CHILD] == 1
        )
        if prox_net_wage_systematic.ndim == 2:
            baby_child = baby_child[:, None]
        non_employment_benefits += calc_erziehungsgeld(
            male_wage=male_wage,
            female_income=non_employment_benefits,
            married=married,
            baby_child=baby_child,
            erziehungsgeld_inc_single=model_spec.erziehungsgeld_income_threshold_single,
            erziehungsgeld_inc_married=model_spec.erziehungsgeld_income_threshold_married,
            erziehungsgeld=model_spec.erziehungsgeld,
        )
    else:
        raise ValueError("Parental leave regime not specified correctly.")

    return non_employment_benefits


def calc_erziehungsgeld(
    male_wage,
    female_income,
    married,
    baby_child,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
):
    relevant_income = male_wage + female_income
    inc_threshold = erziehungsgeld_inc_married * married + erziehungsgeld_inc_single * (
        1 - married
    )
    erz_geld_claim = (relevant_income <= inc_threshold) & baby_child
    return erz_geld_claim * erziehungsgeld


def calculate_alg2(no_child, married, alg2_single, alg_2_alleinerziehend):
    alg2_claim = no_child * alg2_single + (1 - no_child) * alg_2_alleinerziehend
    return alg2_claim * (1 - married)


def calculate_elterngeld(
    hours,
    working_ft_last_period,
    working_pt_last_period,
    prox_net_wage_systematic,
    elterngeld_replacement,
    elterngeld_min,
    elterngeld_max,
    child_benefit,
):
    hours_worked = hours[2] * working_ft_last_period + hours[1] * working_pt_last_period
    elterngeld_claim = working_ft_last_period | working_pt_last_period
    return elterngeld_claim * (
        jnp.minimum(
            jnp.maximum(
                elterngeld_replacement * prox_net_wage_systematic * hours_worked,
                elterngeld_min,
            ),
            elterngeld_max,
        )
        + child_benefit
    )


def calculate_alg1(
    hours,
    working_ft_last_period,
    working_pt_last_period,
    no_child,
    prox_net_wage_systematic,
    alg1_replacement_no_child,
    alg1_replacement_child,
    child_benefit_if_child,
):
    child_benefits = (1 - no_child) * child_benefit_if_child
    replacement_rate = alg1_replacement_no_child * no_child + alg1_replacement_child * (
        1 - no_child
    )

    hours_worked = hours[2] * working_ft_last_period + hours[1] * working_pt_last_period
    alg_1 = working_ft_last_period | working_pt_last_period
    return alg_1 * (
        replacement_rate * prox_net_wage_systematic * hours_worked + child_benefits
    )
