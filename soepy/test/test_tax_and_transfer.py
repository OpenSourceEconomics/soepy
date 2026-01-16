import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import jit

from soepy.pre_processing.tax_and_transfers_params import create_tax_parameters
from soepy.shared.tax_and_transfers import calculate_inc_tax
from soepy.shared.tax_and_transfers import calculate_net_income
from soepy.shared.tax_and_transfers import calculate_ssc_deductions
from soepy.shared.tax_and_transfers_jax import (
    calculate_inc_tax as calculate_inc_tax_jax,
)
from soepy.shared.tax_and_transfers_jax import (
    calculate_net_income as calculate_net_income_jax,
)
from soepy.shared.tax_and_transfers_jax import (
    calculate_ssc_deductions as calculate_ssc_deductions_jax,
)
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


@pytest.fixture(scope="module")
def input_data_jax(input_data):
    """Convert numpy arrays to JAX arrays for JAX tests."""
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    return (
        jnp.array(tax_params),
        jnp.array(deductions_spec),
        thresholds,
        rates,
        intercept_low,
    )


@pytest.mark.parametrize("income", test_incomes)
def test_inc_tax(input_data, income):
    test_pandas = pd.Series(income)

    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data

    soepy_sol = calculate_inc_tax(tax_params, income)

    gettsim_sol = piecewise_polynomial(test_pandas, thresholds, rates, intercept_low)

    np.testing.assert_allclose(soepy_sol, gettsim_sol)


@pytest.mark.parametrize("income", test_incomes)
def test_inc_tax_jax(input_data_jax, income):
    """Test JAX implementation of income tax calculation."""
    test_pandas = pd.Series(income)

    (
        tax_params_jax,
        deductions_spec_jax,
        thresholds,
        rates,
        intercept_low,
    ) = input_data_jax

    soepy_sol_jax = calculate_inc_tax_jax(tax_params_jax, income)

    gettsim_sol = piecewise_polynomial(test_pandas, thresholds, rates, intercept_low)

    np.testing.assert_allclose(soepy_sol_jax, gettsim_sol, rtol=1e-6)


@pytest.mark.parametrize("income", test_incomes)
def test_inc_tax_jax_vs_numba(input_data, input_data_jax, income):
    """Test that JAX and Numba implementations give the same results."""
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    tax_params_jax, deductions_spec_jax, _, _, _ = input_data_jax

    soepy_sol_numba = calculate_inc_tax(tax_params, income)
    soepy_sol_jax = calculate_inc_tax_jax(tax_params_jax, income)

    np.testing.assert_allclose(soepy_sol_numba, soepy_sol_jax, rtol=1e-6)


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
def test_total_tax_and_transfer_jax(input_data_jax, income):
    """Test JAX implementation of net income calculation."""
    (
        tax_params_jax,
        deductions_spec_jax,
        thresholds,
        rates,
        intercept_low,
    ) = input_data_jax
    soli_st = 1.05

    soepy_sol_jax = calculate_net_income_jax(
        tax_params_jax, deductions_spec_jax, income, male_wage=0.0
    )

    gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
        thresholds, rates, intercept_low, np.array(deductions_spec_jax), income, soli_st
    )
    gettsim_sol = taxable_inc - gettsim_tax

    np.testing.assert_allclose(soepy_sol_jax, gettsim_sol, rtol=1e-6)


@pytest.mark.parametrize("income", test_incomes)
def test_total_tax_and_transfer_jax_vs_numba(input_data, input_data_jax, income):
    """Test that JAX and Numba implementations give the same net income."""
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    tax_params_jax, deductions_spec_jax, _, _, _ = input_data_jax

    soepy_sol_numba = calculate_net_income(
        tax_params, deductions_spec, income, male_wage=0
    )
    soepy_sol_jax = calculate_net_income_jax(
        tax_params_jax, deductions_spec_jax, income, male_wage=0.0
    )

    np.testing.assert_allclose(soepy_sol_numba, soepy_sol_jax, rtol=1e-6)


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
def test_ssc_jax(input_data_jax, income):
    """Test JAX implementation of SSC deductions."""
    (
        tax_params_jax,
        deductions_spec_jax,
        thresholds,
        rates,
        intercept_low,
    ) = input_data_jax

    ssc_income = calculate_ssc_deductions_jax(deductions_spec_jax, income)

    deductions_spec = np.array(deductions_spec_jax)
    if income >= deductions_spec[1]:
        ssc_expected = 0.215 * deductions_spec[1]
    else:
        ssc_expected = 0.215 * income
    np.testing.assert_allclose(ssc_income, ssc_expected, rtol=1e-6)


@pytest.mark.parametrize("income", test_incomes)
def test_ssc_jax_vs_numba(input_data, input_data_jax, income):
    """Test that JAX and Numba implementations give the same SSC deductions."""
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    tax_params_jax, deductions_spec_jax, _, _, _ = input_data_jax

    ssc_numba = calculate_ssc_deductions(deductions_spec, income)
    ssc_jax = calculate_ssc_deductions_jax(deductions_spec_jax, income)

    np.testing.assert_allclose(ssc_numba, ssc_jax, rtol=1e-6)


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


@pytest.mark.parametrize("income", test_incomes)
def test_ehegattensplitting_jax(input_data_jax, income):
    """Test JAX implementation of Ehegattensplitting."""
    (
        tax_params_jax,
        deductions_spec_jax,
        thresholds,
        rates,
        intercept_low,
    ) = input_data_jax
    soli_st = 1.05

    partner_ind = np.random.default_rng().binomial(1, 0.5)
    if partner_ind == 1:
        # Calculate gettsim sol on individual basis
        gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
            thresholds,
            rates,
            intercept_low,
            np.array(deductions_spec_jax),
            income / 2,
            soli_st,
        )
        gettsim_sol = (taxable_inc - gettsim_tax) * 2
        soepy_sol_jax = calculate_net_income_jax(
            tax_params_jax, deductions_spec_jax, income / 2, male_wage=income / 2
        )

    else:
        gettsim_tax, taxable_inc = calc_gettsim_sol_individual(
            thresholds,
            rates,
            intercept_low,
            np.array(deductions_spec_jax),
            income,
            soli_st,
        )
        gettsim_sol = taxable_inc - gettsim_tax
        soepy_sol_jax = calculate_net_income_jax(
            tax_params_jax, deductions_spec_jax, income, male_wage=float(partner_ind)
        )

    np.testing.assert_allclose(soepy_sol_jax, gettsim_sol, rtol=1e-6)


@pytest.mark.parametrize("income", test_incomes)
def test_ehegattensplitting_jax_vs_numba(input_data, input_data_jax, income):
    """Test that JAX and Numba implementations give the same results for Ehegattensplitting."""
    tax_params, deductions_spec, thresholds, rates, intercept_low = input_data
    tax_params_jax, deductions_spec_jax, _, _, _ = input_data_jax

    partner_ind = np.random.default_rng().binomial(1, 0.5)

    if partner_ind == 1:
        soepy_sol_numba = calculate_net_income(
            tax_params, deductions_spec, income / 2, male_wage=income / 2
        )
        soepy_sol_jax = calculate_net_income_jax(
            tax_params_jax, deductions_spec_jax, income / 2, male_wage=income / 2
        )
    else:
        soepy_sol_numba = calculate_net_income(
            tax_params, deductions_spec, income, male_wage=partner_ind
        )
        soepy_sol_jax = calculate_net_income_jax(
            tax_params_jax, deductions_spec_jax, income, male_wage=float(partner_ind)
        )

    np.testing.assert_allclose(soepy_sol_numba, soepy_sol_jax, rtol=1e-6)


@pytest.mark.parametrize("income", test_incomes)
def test_jit_compilation(input_data_jax, income):
    """Test that JIT compilation works correctly."""
    tax_params_jax, deductions_spec_jax, _, _, _ = input_data_jax

    # Create JIT-compiled versions
    calculate_net_income_jit = jit(
        calculate_net_income_jax, static_argnames=["tax_splitting"]
    )
    calculate_inc_tax_jit = jit(calculate_inc_tax_jax)
    calculate_ssc_jit = jit(calculate_ssc_deductions_jax)

    # Test that JIT versions give same results as non-JIT
    net_income = calculate_net_income_jax(
        tax_params_jax, deductions_spec_jax, income, male_wage=0.0
    )
    net_income_jit = calculate_net_income_jit(
        tax_params_jax, deductions_spec_jax, income, male_wage=0.0
    )

    inc_tax = calculate_inc_tax_jax(tax_params_jax, income)
    inc_tax_jit = calculate_inc_tax_jit(tax_params_jax, income)

    ssc = calculate_ssc_deductions_jax(deductions_spec_jax, income)
    ssc_jit = calculate_ssc_jit(deductions_spec_jax, income)

    np.testing.assert_allclose(net_income, net_income_jit, rtol=1e-6)
    np.testing.assert_allclose(inc_tax, inc_tax_jit, rtol=1e-6)
    np.testing.assert_allclose(ssc, ssc_jit, rtol=1e-6)
