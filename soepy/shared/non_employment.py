import numba
import numpy as np

from soepy.shared.shared_constants import HOURS
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
):
    """This function calculates the non employment consumption resources. It first
    calcultes the non employment benefits before using them to calculate the resources."""

    alg1_replacement_no_child = model_spec.alg1_replacement_no_child
    alg1_replacement_child = model_spec.alg1_replacement_child
    regelsatz_single = model_spec.regelsatz_single
    housing_single = model_spec.housing_single
    housing_addtion = model_spec.housing_addtion
    regelsatz_child = model_spec.regelsatz_child
    addition_child_single = model_spec.addition_child_single
    elterngeld_replacement = model_spec.elterngeld_replacement
    elterngeld_min = model_spec.elterngeld_min
    elterngeld_max = model_spec.elterngeld_max

    if model_spec.parental_leave_regime == "elterngeld":
        elterngeld_regime = True
    elif model_spec.parental_leave_regime == "erziehungsgeld":
        elterngeld_regime = False
    else:
        raise ValueError("Parental leave regime not specified correctly.")

    erziehungsgeld_inc_single = model_spec.erziehungsgeld_income_threshold_single
    erziehungsgeld_inc_married = model_spec.erziehungsgeld_income_threshold_married
    erziehungsgeld = model_spec.erziehungsgeld

    return calc_resources(
        deductions_spec,
        income_tax_spec,
        np.array(HOURS),
        states,
        log_wage_systematic,
        male_wage,
        child_benefits,
        alg1_replacement_no_child,
        alg1_replacement_child,
        regelsatz_single,
        housing_single,
        housing_addtion,
        regelsatz_child,
        addition_child_single,
        elterngeld_replacement,
        elterngeld_min,
        elterngeld_max,
        erziehungsgeld_inc_single,
        erziehungsgeld_inc_married,
        erziehungsgeld,
        tax_splitting,
        elterngeld_regime,
    )


def calculate_non_employment_benefits(
    hours,
    states,
    log_wage_systematic,
    child_benefit,
    male_wage,
    alg1_replacement_no_child,
    alg1_replacement_child,
    regelsatz_single,
    housing_single,
    housing_addtion,
    regelsatz_child,
    addition_child_single,
    elterngeld_replacement,
    elterngeld_min,
    elterngeld_max,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
    elterngeld_regime,
):
    """This function calculates the benefits an individual would receive if they were
    to choose to be non-employed in the period"""

    no_child = states[:, 6] == -1
    working_ft_last_period = states[:, 2] == 2
    working_pt_last_period = states[:, 2] == 1
    married = states[:, 7] == 1

    prox_net_wage_systematic = 0.65 * np.exp(log_wage_systematic)

    alg2_single = regelsatz_single + housing_single

    alg_2_alleinerziehend = (
        regelsatz_single
        + regelsatz_child
        + addition_child_single
        + housing_single
        + housing_addtion
    )

    alg2 = calculate_alg2(
        no_child,
        married,
        alg2_single,
        alg_2_alleinerziehend,
    )
    if elterngeld_regime:
        newborn_child = states[:, 6] == 0

        last_working_non_employment_benefits = np.zeros(states.shape[0])

        elterngeld = calculate_elterngeld(
            hours,
            working_ft_last_period,
            working_pt_last_period,
            prox_net_wage_systematic,
            elterngeld_replacement,
            elterngeld_min,
            elterngeld_max,
            child_benefit,
        )

        alg1 = calculate_alg1(
            hours,
            working_ft_last_period,
            working_pt_last_period,
            no_child,
            prox_net_wage_systematic,
            alg1_replacement_no_child,
            alg1_replacement_child,
            child_benefit,
        )

        last_working_non_employment_benefits = (
            1 - newborn_child
        ) * alg1 + newborn_child * elterngeld

        non_employment_benefits = np.maximum(
            last_working_non_employment_benefits,
            alg2,
        )
    else:
        non_employment_benefits = np.maximum(
            calculate_alg1(
                hours,
                working_ft_last_period,
                working_pt_last_period,
                no_child,
                prox_net_wage_systematic,
                alg1_replacement_no_child,
                alg1_replacement_child,
                child_benefit,
            ),
            alg2,
        )
        baby_child = (states[:, 6] == 0) | (states[:, 6] == 1)
        non_employment_benefits += calc_erziehungsgeld(
            male_wage,
            non_employment_benefits,
            married,
            baby_child,
            erziehungsgeld_inc_single,
            erziehungsgeld_inc_married,
            erziehungsgeld,
        )

    return non_employment_benefits


@numba.njit
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


def calculate_alg2(
    no_child,
    married,
    alg2_single,
    alg_2_alleinerziehend,
):
    # All partners work full-time so there exists only a claim if not married.
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
    """This implements the 2007 elterngeld regime."""
    hours_worked = hours[2] * working_ft_last_period + hours[1] * working_pt_last_period
    elterngeld_claim = working_ft_last_period | working_pt_last_period
    return elterngeld_claim * (
        np.minimum(
            np.maximum(
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

    """Individual worked last period: ALG I based on labor income the individual
    would have earned working full-time in the period (excluding wage shock)
    for a person who worked last period 60% if no child"""
    child_benefits = (1 - no_child) * child_benefit_if_child
    replacement_rate = alg1_replacement_no_child * no_child + alg1_replacement_child * (
        1 - no_child
    )

    hours_worked = hours[2] * working_ft_last_period + hours[1] * working_pt_last_period
    alg_1 = working_ft_last_period | working_pt_last_period
    return alg_1 * (
        replacement_rate * prox_net_wage_systematic * hours_worked + child_benefits
    )


def calc_resources(
    deductions_spec,
    income_tax_spec,
    hours,
    state,
    log_wage_systematic,
    male_wage,
    child_benefit,
    alg1_replacement_no_child,
    alg1_replacement_child,
    regelsatz_single,
    housing_single,
    housing_addtion,
    regelsatz_child,
    addition_child_single,
    elterngeld_replacement,
    elterngeld_min,
    elterngeld_max,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
    tax_splitting,
    elterngeld_regime,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to not be employed.
    It adds the components from the budget constraint to the female wage."""

    non_employment_benefits = calculate_non_employment_benefits(
        hours,
        state,
        log_wage_systematic,
        child_benefit,
        male_wage,
        alg1_replacement_no_child,
        alg1_replacement_child,
        regelsatz_single,
        housing_single,
        housing_addtion,
        regelsatz_child,
        addition_child_single,
        elterngeld_replacement,
        elterngeld_min,
        elterngeld_max,
        erziehungsgeld_inc_single,
        erziehungsgeld_inc_married,
        erziehungsgeld,
        elterngeld_regime,
    )

    # Set female wage to 0
    male_net_income = calculate_net_income(
        income_tax_spec, deductions_spec, 0, male_wage, tax_splitting
    )

    return male_net_income + non_employment_benefits
