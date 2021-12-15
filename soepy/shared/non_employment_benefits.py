import numpy as np

from soepy.shared.shared_constants import HOURS


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

    non_employment_benefits[:, 0] = calculate_alg1(
        working_ft_last_period,
        working_pt_last_period,
        no_child,
        newborn_child,
        prox_net_wage_systematic,
        model_spec.alg1_replacement_no_child,
        model_spec.alg1_replacement_child,
    )

    non_employment_benefits[:, 1] = calculate_alg2(
        working_last_period,
        no_child,
        married,
        model_spec.regelsatz_single,
        model_spec.housing,
        model_spec.regelsatz_partner,
        model_spec.regelsatz_child,
        model_spec.addition_child_single,
    )

    non_employment_benefits[:, 2] = calculate_elterngeld(
        working_ft_last_period,
        working_pt_last_period,
        newborn_child,
        prox_net_wage_systematic,
        model_spec.motherhood_replacement,
    )

    return non_employment_benefits


def calculate_alg2(
    working_last_period,
    no_child,
    married,
    regelsatz_single,
    housing,
    regelsatz_partner,
    regelsatz_child,
    addition_child_single,
):
    # Individual did not work last period: Social assistance
    # No partner, No child
    alg2 = np.where(
        (~working_last_period & no_child & ~married),
        regelsatz_single + housing * 1.25,
        0.00,
    )
    # Yes partner, No child
    alg2 = np.where(
        (~working_last_period & no_child & married),
        2 * regelsatz_partner + housing * 1.5,
        alg2,
    )
    # Yes partner, Yes child
    alg2 = np.where(
        (~working_last_period & ~no_child & married),
        2 * regelsatz_partner + regelsatz_child + housing * 1.5,
        alg2,
    )
    # No partner, Yes child
    alg2 = np.where(
        (~working_last_period & ~no_child & ~married),
        regelsatz_single + regelsatz_child + addition_child_single + housing * 0.25,
        alg2,
    )
    return alg2


def calculate_elterngeld(
    working_ft_last_period,
    working_pt_last_period,
    newborn_child,
    prox_net_wage_systematic,
    motherhood_replacement,
):
    """This implements the 2007 elterngeld regime."""
    elterngeld = np.where(
        (working_ft_last_period & newborn_child),
        (motherhood_replacement * prox_net_wage_systematic * HOURS[2]).clip(
            min=300, max=1800
        ),
        0.00,
    )
    elterngeld = np.where(
        (working_pt_last_period & newborn_child),
        (motherhood_replacement * prox_net_wage_systematic * HOURS[1]).clip(
            min=300, max=1800
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
