import numba
import numpy as np

from soepy.shared.shared_constants import HOURS
from soepy.shared.tax_and_transfers import calculate_net_income


def calculate_non_employment_benefits(model_spec, states, log_wage_systematic):
    """This function calculates the benefits an individual would receive if they were
    to choose to be non-employed in the period"""

    non_employment_benefits = np.full((states.shape[0], 3), np.nan)
    no_child = states[:, 6] == -1
    newborn_child = states[:, 6] == 0
    working_ft_last_period = states[:, 2] == 2
    working_pt_last_period = states[:, 2] == 1
    working_last_period = working_ft_last_period | working_pt_last_period
    married = states[:, 7] == 1

    prox_net_wage_systematic = 0.65 * np.exp(log_wage_systematic)

    alg1_replacement_no_child = model_spec.alg1_replacement_no_child
    alg1_replacement_child = model_spec.alg1_replacement_child
    regelsatz_single = model_spec.regelsatz_single
    housing_single = model_spec.housing_single
    housing_addtion = model_spec.housing_addtion
    regelsatz_child = model_spec.regelsatz_child
    addition_child_single = model_spec.addition_child_single
    motherhood_replacement = model_spec.motherhood_replacement
    elterngeld_min = model_spec.elterngeld_min
    elterngeld_max = model_spec.elterngeld_max

    alg2_single = regelsatz_single + housing_single
    alg_2_alleinerziehend = (
        regelsatz_single
        + regelsatz_child
        + addition_child_single
        + housing_single
        + housing_addtion
    )

    non_employment_benefits[:, 0] = calculate_alg1(
        working_ft_last_period,
        working_pt_last_period,
        no_child,
        newborn_child,
        prox_net_wage_systematic,
        alg1_replacement_no_child,
        alg1_replacement_child,
    )

    non_employment_benefits[:, 1] = calculate_alg2(
        working_last_period, no_child, married, alg2_single, alg_2_alleinerziehend
    )

    non_employment_benefits[:, 2] = calculate_elterngeld(
        working_ft_last_period,
        working_pt_last_period,
        newborn_child,
        prox_net_wage_systematic,
        motherhood_replacement,
        elterngeld_min,
        elterngeld_max,
    )

    return non_employment_benefits


def calculate_alg2(
    working_last_period, no_child, married, alg2_single, alg_2_alleinerziehend
):
    # Individual did not work last period: Social assistance if not married.
    # No child:
    alg2 = np.where(
        (~working_last_period & no_child & ~married),
        alg2_single,
        0.00,
    )
    # Yes child:
    alg2 = np.where(
        (~working_last_period & ~no_child & ~married),
        alg_2_alleinerziehend,
        alg2,
    )
    return alg2


def calculate_elterngeld(
    working_ft_last_period,
    working_pt_last_period,
    newborn_child,
    prox_net_wage_systematic,
    motherhood_replacement,
    elterngeld_min,
    elterngeld_max,
):
    """This implements the 2007 elterngeld regime."""
    elterngeld = np.where(
        (working_ft_last_period & newborn_child),
        (motherhood_replacement * prox_net_wage_systematic * HOURS[2]).clip(
            min=elterngeld_min, max=elterngeld_max
        ),
        0.00,
    )
    elterngeld = np.where(
        (working_pt_last_period & newborn_child),
        (motherhood_replacement * prox_net_wage_systematic * HOURS[1]).clip(
            min=elterngeld_min, max=elterngeld_max
        ),
        elterngeld,
    )
    return elterngeld


def calculate_alg1(
    working_ft_last_period,
    working_pt_last_period,
    no_child,
    newborn_child,
    prox_net_wage_systematic,
    alg1_replacement_no_child,
    alg1_replacement_child,
):

    """Individual worked last period: ALG I based on labor income the individual
    would have earned working full-time in the period (excluding wage shock)
    for a person who worked last period 60% if no child"""
    alg1 = np.where(
        (working_ft_last_period & no_child),
        alg1_replacement_no_child * prox_net_wage_systematic * HOURS[2],
        0.00,
    )
    alg1 = np.where(
        (working_pt_last_period & no_child),
        alg1_replacement_no_child * prox_net_wage_systematic * HOURS[1],
        alg1,
    )
    # 67% if child
    alg1 = np.where(
        (working_ft_last_period & ~no_child & ~newborn_child),
        alg1_replacement_child * prox_net_wage_systematic * HOURS[2],
        alg1,
    )
    alg1 = np.where(
        (working_pt_last_period & ~no_child & ~newborn_child),
        alg1_replacement_child * prox_net_wage_systematic * HOURS[1],
        alg1,
    )
    return alg1


@numba.guvectorize(
    ["f8[:], f8[:, :], f8, f8[:], b1, f8[:]"],
    "(n_ssc_params), (n_tax_params, n_tax_params), (), (n_choices), () -> ()",
    nopython=True,
    target="cpu",
    # target="parallel",
)
def calculate_non_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    male_wage,
    non_employment_benefits,
    tax_splitting,
    non_employment_consumption_resources,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to not be employed.
    It adds the components from the budget constraint to the female wage."""

    # Set female wage to
    net_income = (
        calculate_net_income(
            income_tax_spec, deductions_spec, 0, male_wage, tax_splitting
        )
        + non_employment_benefits[0]
    )

    non_employment_consumption_resources[0] = (
        net_income + non_employment_benefits[1] + non_employment_benefits[2]
    )
