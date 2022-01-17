import numpy as np
import pandas as pd
import pytest

from soepy.pre_processing.tax_and_transfers_params import create_tax_parameters
from soepy.shared.tax_and_transfers import calculate_inc_tax
from soepy.shared.tax_and_transfers import calculate_net_income
from soepy.shared.tax_and_transfers import calculate_ssc_deductions
from soepy.test.resources.gettsim_tax_func import calc_gettsim_sol_individual
from soepy.test.resources.gettsim_tax_func import piecewise_polynomial

test_incomes = np.arange(1, 2000, 100).astype(float)


@pytest.fixture(scope="module")
def input_data():
    tax_params = create_tax_parameters()
    deductions_spec = np.array([0.215, 63_000 / (12 * 4.3)])
    thresholds = np.append(np.append(-np.inf, tax_params[0, :]), np.inf)
    # In gettsim the tax function takes the linear rates and the
    # quadratic rates (progressionsfactor)
    rates = np.empty((2, 5))
    rates[0, 0] = 0
    rates[0, 1:] = tax_params[2, :]
    rates[1, :] = 0
    rates[1, 1:] = tax_params[3, :]

    intercept_low = np.append(0, tax_params[1, :])
    return tax_params, deductions_spec, thresholds, rates, intercept_low


@pytest.mark.parametrize("income", test_incomes)
def test_inc_tax(input_data, income):
    test_pandas = pd.Series(income)

    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data

    soepy_sol = calculate_inc_tax(tax_params, income)

    gettsim_sol = piecewise_polynomial(test_pandas, thresholds, rates, intercept_low)

    np.testing.assert_allclose(soepy_sol, gettsim_sol)


@pytest.mark.parametrize("income", test_incomes)
def test_total_tax_and_transfer(input_data, income):
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    soli_st = 1.05

    soepy_sol = calculate_net_income(tax_params, deductions_spec, income, male_wage=0)

    gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
        thresholds, rates, intercept_low, deductions_spec, income, soli_st
    )
    gettsim_sol = taxable_inc - gettsim_tax

    np.testing.assert_allclose(soepy_sol, gettsim_sol)


@pytest.mark.parametrize("income", test_incomes)
def test_ssc(input_data, income):
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data

    ssc_income = calculate_ssc_deductions(deductions_spec, income)

    if income >= deductions_spec[1]:
        ssc_expected = calculate_ssc_deductions(deductions_spec, deductions_spec[1])
    else:
        ssc_expected = 0.215 * income
    np.testing.assert_allclose(ssc_income, ssc_expected)


@pytest.mark.parametrize("income", test_incomes)
def test_ehegattensplitting(input_data, income):
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    soli_st = 1.05

    partner_ind = np.random.default_rng().binomial(1, 0.5)
    if partner_ind == 1:
        # Calculate gettsim sol on individual basis
        gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
            thresholds, rates, intercept_low, deductions_spec, income / 2, soli_st
        )
        gettsim_sol = (taxable_inc - gettsim_tax) * 2
        soepy_sol = calculate_net_income(
            tax_params, deductions_spec, income / 2, male_wage=income / 2
        )

    else:
        gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
            thresholds, rates, intercept_low, deductions_spec, income, soli_st
        )
        gettsim_sol = taxable_inc - gettsim_tax
        soepy_sol = calculate_net_income(
            tax_params, deductions_spec, income, male_wage=partner_ind
        )

    np.testing.assert_allclose(soepy_sol, gettsim_sol)
