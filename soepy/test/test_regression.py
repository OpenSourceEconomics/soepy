import pickle
import random

import numpy as np

from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR


def test1():
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """

    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    for i in random.sample(range(0, 100), 10):

        model_spec_init_dict, random_model_params_df, expected_df = tests[i]

        model_spec_init_dict["EXOG_PROC"]["kids_info_file_name"] = (
            str(TEST_RESOURCES_DIR) + "/" + "exog_child_info.pkl"
        )
        model_spec_init_dict["EXOG_PROC"]["eud_info_file_name"] = (
            str(TEST_RESOURCES_DIR) + "/" + "exog_educ_info_generic.pkl"
        )

        calculated_df = simulate(random_model_params_df, model_spec_init_dict)

        for col in expected_df.columns.tolist():
            np.testing.assert_array_almost_equal(
                expected_df[col], calculated_df[col],
            )
