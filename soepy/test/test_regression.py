import pytest
import pickle
import random

import numpy as np

from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from development.tests.auxiliary.auxiliary import cleanup


@pytest.mark.skip(reason="not working, ideas why?")
def test1():
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """

    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    for i in random.sample(range(0, 100), 10):

        (
            model_spec_init_dict,
            random_model_params_df,
            exog_child_info,
            exog_educ_info,
            expected_df,
        ) = tests[i]

        model_spec_init_dict["EXOG_PROC"]["partner_info_file_name"] = "test"

        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_educ_info.to_pickle("test.soepy.educ.pkl")

        calculated_df = simulate(random_model_params_df, model_spec_init_dict)

        for col in expected_df.columns.tolist():
            print(col)
            np.testing.assert_array_almost_equal(
                expected_df[col], calculated_df[col],
            )

    cleanup()
