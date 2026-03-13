import collections

import numpy as np
import pytest

from soepy.shared.non_employment import calculate_non_employment_benefits


@pytest.fixture()
def model_spec():
    spec = {
        "alg1_replacement_no_child": 0.6,
        "alg1_replacement_child": 0.67,
        "regelsatz_single": 10.0,
        "housing_single": 5.0,
        "housing_addtion": 1.0,
        "regelsatz_child": 4.0,
        "addition_child_single": 2.0,
        "elterngeld_replacement": 0.67,
        "elterngeld_min": 1.0,
        "elterngeld_max": 20.0,
        "erziehungsgeld_income_threshold_single": 999.0,
        "erziehungsgeld_income_threshold_married": 999.0,
        "erziehungsgeld": 0.0,
        "parental_leave_regime": "elterngeld",
    }
    return collections.namedtuple("model_specification", spec.keys())(**spec)


def test_non_employment_benefits_smoke(model_spec):

    n = 5
    states = np.zeros((n, 6), dtype=int)

    # lagged_choice
    states[:, 2] = 0

    # age_youngest_child
    states[:, 4] = -1

    # partner
    states[:, 5] = 0

    out = calculate_non_employment_benefits(
        hours=np.array([0, 18, 38]),
        states=states,
        log_wage_systematic=np.zeros(n),
        child_benefit=np.zeros(n),
        male_wage=np.zeros(n),
        income_tax_spec=np.zeros((4, 4)),
        deductions_spec=np.zeros(2),
        tax_splitting=False,
        model_spec=model_spec,
    )

    assert out.shape == (n,)


def test_non_employment_benefits_use_female_net_income_only():
    spec = {
        "alg1_replacement_no_child": 0.5,
        "alg1_replacement_child": 0.5,
        "regelsatz_single": 0.0,
        "housing_single": 0.0,
        "housing_addtion": 0.0,
        "regelsatz_child": 0.0,
        "addition_child_single": 0.0,
        "elterngeld_replacement": 0.5,
        "elterngeld_min": 0.0,
        "elterngeld_max": 1000.0,
        "erziehungsgeld_income_threshold_single": 999.0,
        "erziehungsgeld_income_threshold_married": 999.0,
        "erziehungsgeld": 0.0,
        "parental_leave_regime": "elterngeld",
    }
    model_spec = collections.namedtuple("model_specification", spec.keys())(**spec)

    states = np.zeros((1, 6), dtype=int)
    states[:, 2] = 1
    states[:, 4] = -1
    states[:, 5] = 0

    hours = np.array([0, 20, 40])
    income_tax_spec = np.zeros((4, 4))
    deductions_spec = np.zeros(2)

    out_no_male = calculate_non_employment_benefits(
        hours=hours,
        states=states,
        log_wage_systematic=np.zeros(1),
        child_benefit=np.zeros(1),
        male_wage=np.zeros(1),
        income_tax_spec=income_tax_spec,
        deductions_spec=deductions_spec,
        tax_splitting=False,
        model_spec=model_spec,
    )

    out_with_male = calculate_non_employment_benefits(
        hours=hours,
        states=states,
        log_wage_systematic=np.zeros(1),
        child_benefit=np.zeros(1),
        male_wage=np.array([1000.0]),
        income_tax_spec=income_tax_spec,
        deductions_spec=deductions_spec,
        tax_splitting=False,
        model_spec=model_spec,
    )

    expected = np.array([0.5 * hours[1]])
    np.testing.assert_allclose(out_no_male, expected)
    np.testing.assert_allclose(out_with_male, expected)


def test_non_employment_benefits_support_experience_grid_and_ignore_male_wage():
    spec = {
        "alg1_replacement_no_child": 0.5,
        "alg1_replacement_child": 0.5,
        "regelsatz_single": 0.0,
        "housing_single": 0.0,
        "housing_addtion": 0.0,
        "regelsatz_child": 0.0,
        "addition_child_single": 0.0,
        "elterngeld_replacement": 0.5,
        "elterngeld_min": 0.0,
        "elterngeld_max": 1e9,
        "erziehungsgeld_income_threshold_single": 999.0,
        "erziehungsgeld_income_threshold_married": 999.0,
        "erziehungsgeld": 0.0,
        "parental_leave_regime": "elterngeld",
    }
    model_spec = collections.namedtuple("model_specification", spec.keys())(**spec)

    n_states, n_grid = 3, 4

    states = np.zeros((n_states, 6), dtype=int)
    states[:, 4] = -1
    states[:, 5] = 0
    states[0, 2] = 2
    states[1, 2] = 1
    states[2, 2] = 0

    hours = np.array([0, 20, 40])
    log_wage_grid = np.zeros((n_states, n_grid))
    income_tax_spec = np.zeros((4, 4))
    deductions_spec = np.zeros(2)

    out_no_male = calculate_non_employment_benefits(
        hours=hours,
        states=states,
        log_wage_systematic=log_wage_grid,
        child_benefit=np.zeros(n_states),
        male_wage=np.zeros(n_states),
        income_tax_spec=income_tax_spec,
        deductions_spec=deductions_spec,
        tax_splitting=False,
        model_spec=model_spec,
    )

    out_with_male = calculate_non_employment_benefits(
        hours=hours,
        states=states,
        log_wage_systematic=log_wage_grid,
        child_benefit=np.zeros(n_states),
        male_wage=np.array([1000.0, 1000.0, 1000.0]),
        income_tax_spec=income_tax_spec,
        deductions_spec=deductions_spec,
        tax_splitting=False,
        model_spec=model_spec,
    )

    assert out_no_male.shape == (n_states, n_grid)
    np.testing.assert_allclose(out_no_male, out_with_male)

    expected = np.array([[20.0] * n_grid, [10.0] * n_grid, [0.0] * n_grid])
    np.testing.assert_allclose(out_no_male, expected)
