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

        # Remove after successful testing
        const_p = random_model_params_df.loc[("disutil_work", "const_p")]["value"]
        const_f = random_model_params_df.loc[("disutil_work", "const_f")]["value"]

        random_model_params_df.loc[("disutil_work", "no_kids_f"), "value"] = const_f
        random_model_params_df.loc[("disutil_work", "no_kids_p"), "value"] = (
            const_p - const_f
        )
        random_model_params_df.loc[("disutil_work", "yes_kids_f"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "yes_kids_p"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_02_f"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_02_p"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_35_f"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_35_p"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_610_f"), "value"] = 0.00
        random_model_params_df.loc[("disutil_work", "child_610_p"), "value"] = 0.00

        calculated_df = simulate(random_model_params_df, model_spec_init_dict)
        print(sum(calculated_df["Age_Youngest_Child"] != -1))

        for col in expected_df.columns.tolist():
            np.testing.assert_array_almost_equal(
                expected_df[col][expected_df[col].notna()],
                calculated_df[col][calculated_df[col].notna()],
            )
