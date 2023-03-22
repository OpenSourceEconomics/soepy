import numba
import numpy as np

from soepy.shared.shared_constants import HOURS
from soepy.shared.tax_and_transfers import calculate_net_income


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


@numba.njit(nogil=True)
def calculate_non_employment_benefits(
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
):
    """This function calculates the benefits an individual would receive if they were
    to choose to be non-employed in the period"""

    non_employment_benefits = np.full((3,), np.nan)
    no_child = state[6] == -1
    working_ft_last_period = state[2] == 2
    working_pt_last_period = state[2] == 1
    working_last_period = working_ft_last_period | working_pt_last_period
    married = state[7] == 1

    prox_net_wage_systematic = 0.65 * np.exp(log_wage_systematic)

    alg2_single = regelsatz_single + housing_single

    alg_2_alleinerziehend = (
        regelsatz_single
        + regelsatz_child
        + addition_child_single
        + housing_single
        + housing_addtion
    )

    non_employment_benefits[1] = calculate_alg2(
        working_last_period,
        no_child,
        married,
        alg2_single,
        alg_2_alleinerziehend,
    )

    if elterngeld_regime:
        newborn_child = state[6] == 0

        if newborn_child:
            non_employment_benefits[0] = 0

            non_employment_benefits[2] = calculate_elterngeld(
                hours,
                working_ft_last_period,
                working_pt_last_period,
                prox_net_wage_systematic,
                elterngeld_replacement,
                elterngeld_min,
                elterngeld_max,
                child_benefit,
            )
        else:
            non_employment_benefits[0] = calculate_alg1(
                hours,
                working_ft_last_period,
                working_pt_last_period,
                no_child,
                prox_net_wage_systematic,
                alg1_replacement_no_child,
                alg1_replacement_child,
                child_benefit,
            )
            non_employment_benefits[2] = 0

    else:
        non_employment_benefits[0] = calculate_alg1(
            hours,
            working_ft_last_period,
            working_pt_last_period,
            no_child,
            prox_net_wage_systematic,
            alg1_replacement_no_child,
            alg1_replacement_child,
            child_benefit,
        )
        baby_child = (state[6] == 0) | (state[6] == 1)
        non_employment_benefits[2] = calc_erziehungsgeld(
            male_wage,
            non_employment_benefits[0],
            married,
            baby_child,
            erziehungsgeld_inc_single,
            erziehungsgeld_inc_married,
            erziehungsgeld,
        )

    return non_employment_benefits


@numba.njit(nogil=True)
def calc_erziehungsgeld(
    male_wage,
    alg1,
    married,
    baby_child,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
):
    relevant_income = male_wage + alg1
    if married:
        if (relevant_income <= erziehungsgeld_inc_married) & baby_child:
            return erziehungsgeld
        else:
            return 0
    else:
        if (relevant_income <= erziehungsgeld_inc_single) & baby_child:
            return erziehungsgeld
        else:
            return 0


@numba.njit(nogil=True)
def calculate_alg2(
    working_last_period,
    no_child,
    married,
    alg2_single,
    alg_2_alleinerziehend,
):
    # Individual did not work last period: Social assistance if not married.

    # No child:
    if ~working_last_period & no_child & ~married:
        return alg2_single
    # Has a child. We deduct the child benefit as it is added for all three unemployment
    # benefits in the last step and you don't get it in alg2.
    elif ~working_last_period & ~no_child & ~married:
        return alg_2_alleinerziehend
    else:
        return 0


@numba.njit(nogil=True)
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
    if working_ft_last_period:
        return (
            np.minimum(
                np.maximum(
                    elterngeld_replacement * prox_net_wage_systematic * hours[2],
                    elterngeld_min,
                ),
                elterngeld_max,
            )
            + child_benefit
        )
    elif working_pt_last_period:
        return (
            np.minimum(
                np.maximum(
                    elterngeld_replacement * prox_net_wage_systematic * hours[1],
                    elterngeld_min,
                ),
                elterngeld_max,
            )
            + child_benefit
        )
    else:
        return 0


@numba.njit(nogil=True)
def calculate_alg1(
    hours,
    working_ft_last_period,
    working_pt_last_period,
    no_child,
    prox_net_wage_systematic,
    alg1_replacement_no_child,
    alg1_replacement_child,
    child_benefit,
):

    """Individual worked last period: ALG I based on labor income the individual
    would have earned working full-time in the period (excluding wage shock)
    for a person who worked last period 60% if no child"""
    if working_ft_last_period & no_child:
        return alg1_replacement_no_child * prox_net_wage_systematic * hours[2]
    elif working_pt_last_period & no_child:
        return alg1_replacement_no_child * prox_net_wage_systematic * hours[1]

    # 67% if child
    elif working_ft_last_period & ~no_child:
        return (
            alg1_replacement_child * prox_net_wage_systematic * hours[2] + child_benefit
        )
    elif working_pt_last_period & ~no_child:
        return (
            alg1_replacement_child * prox_net_wage_systematic * hours[1] + child_benefit
        )
    else:
        return 0


@numba.guvectorize(
    [
        "f8[:], f8[:, :], f8[:], f8[:], f8, f8, f8, "
        "f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8, b1, b1, f8[:]"
    ],
    "(n_ssc_params), (n_tax_params, n_tax_params), (n_choices), (n_state_vars), (), (),"
    " (), (),(),(),(),(),(),(),(),(),(),(), (),(),(), () -> ()",
    nopython=True,
    target="cpu",
    # target="parallel",
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
    non_employment_consumption_resources,
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

    non_employment_consumption_resources[0] = (
        male_net_income
        + non_employment_benefits[0]
        + non_employment_benefits[1]
        + non_employment_benefits[2]
    )
