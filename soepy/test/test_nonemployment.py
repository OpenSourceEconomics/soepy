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
        alg1_replacement_no_child=model_spec.alg1_replacement_no_child,
        alg1_replacement_child=model_spec.alg1_replacement_child,
        regelsatz_single=model_spec.regelsatz_single,
        housing_single=model_spec.housing_single,
        housing_addtion=model_spec.housing_addtion,
        regelsatz_child=model_spec.regelsatz_child,
        addition_child_single=model_spec.addition_child_single,
        elterngeld_replacement=model_spec.elterngeld_replacement,
        elterngeld_min=model_spec.elterngeld_min,
        elterngeld_max=model_spec.elterngeld_max,
        erziehungsgeld_inc_single=model_spec.erziehungsgeld_income_threshold_single,
        erziehungsgeld_inc_married=model_spec.erziehungsgeld_income_threshold_married,
        erziehungsgeld=model_spec.erziehungsgeld,
        elterngeld_regime=True,
    )

    assert out.shape == (n,)
