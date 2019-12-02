import pickle
import numpy as np
import pytest
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR


@pytest.mark.parametrize("idx", range(10))
def test1(idx):
    """This test runs a random selection of five regression tests from
    our regression test battery.
    """

    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    test = tests[idx]

    model_spec_init_dict, random_model_params_df, expected_df = test

    model_spec_init_dict["EDUC_LEVEL_BOUNDS"] = dict()
    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["low_bound"] = 10
    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["middle_bound"] = 11
    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["high_bound"] = 12

    calculated_df = simulate(random_model_params_df, model_spec_init_dict)

    for col in expected_df.columns.tolist():
        np.testing.assert_array_almost_equal(
            expected_df[col][expected_df[col].notna()],
            calculated_df[col][calculated_df[col].notna()],
        )
