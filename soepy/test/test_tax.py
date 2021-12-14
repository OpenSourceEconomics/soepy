from soepy.test.resources.gettsim_tax_func import piecewise_polynomial
from soepy.shared.tax_and_transfers import calculate_tax, calculate_inc_tax
import numpy as np
import pandas as pd
import pytest

test_incomes = np.arange(1, 2000, 100).astype(float)


@pytest.fixture(scope="module")
def input_data():
    tax_params = np.array(
        [
            163.00,
            1001.00,
            0.14,
            0.42,
            0.055,
        ]
    )
    thresholds = np.array([-np.inf, tax_params[0], tax_params[1], np.inf])
    # In gettsim the tax function takes the linear rates and the
    # quadratic rates (progressionsfactor)
    rates = np.empty((2, 3))
    rates[0, 0] = 0
    rates[0, 1:] = tax_params[2:4]
    rates[1, :] = 0
    rates[1, 1] = (tax_params[3] - tax_params[2]) / (tax_params[1] - tax_params[0]) / 2

    intercept_low = np.array([0, 0, calculate_inc_tax(tax_params, tax_params[1])])
    return tax_params, thresholds, rates, intercept_low


@pytest.mark.parametrize("income", test_incomes)
def test_inc_tax(input_data, income):
    test_pandas = pd.Series(income)

    tax_params, thresholds, rates, intercept_low = input_data

    soepy_sol = calculate_inc_tax(tax_params, income)

    gettsim_sol = piecewise_polynomial(test_pandas, thresholds, rates, intercept_low)

    np.testing.assert_allclose(soepy_sol, gettsim_sol)


@pytest.mark.parametrize("income", test_incomes)
def test_total_tax(input_data, income):
    test_pandas = pd.Series(income)

    tax_params, thresholds, rates, intercept_low = input_data
    soli_st = 1 + tax_params[-1]

    soepy_sol = calculate_tax(tax_params, income, male_wage=0)

    gettsim_sol = piecewise_polynomial(test_pandas, thresholds, rates, intercept_low)

    np.testing.assert_allclose(soepy_sol, gettsim_sol * soli_st)


@pytest.mark.parametrize("income", test_incomes)
def test_ehegattensplitting(input_data, income):
    tax_params, thresholds, rates, intercept_low = input_data
    soli_st = 1 + tax_params[-1]

    partner_ind = np.random.default_rng().binomial(1, 0.5)
    if partner_ind == 1:
        gettsim_sol = piecewise_polynomial(
            pd.Series([income / 2, income / 2]), thresholds, rates, intercept_low
        ).sum()
        soepy_sol = calculate_tax(tax_params, income, male_wage=income / 2)

    else:
        gettsim_sol = piecewise_polynomial(
            pd.Series(income), thresholds, rates, intercept_low
        )
        soepy_sol = calculate_tax(tax_params, income, male_wage=partner_ind)

    np.testing.assert_allclose(soepy_sol, gettsim_sol * soli_st)
