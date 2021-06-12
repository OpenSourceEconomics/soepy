import pickle
import random

import numpy as np

from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from development.tests.auxiliary.auxiliary import cleanup


def test1():
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """

    vault = TEST_RESOURCES_DIR / "regression_vault.soepy_.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    for i in random.sample(range(0, 100), 10):

        (
            model_spec_init_dict,
            random_model_params_df,
            exog_educ_shares,
            exog_child_age_shares,
            exog_partner_shares,
            exog_child_info,
            exog_partner_arrival_info,
            exog_partner_separation_info,
            expected_df,
        ) = tests[i]

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

        calculated_df = simulate(random_model_params_df, model_spec_init_dict)

        for col in expected_df.columns.tolist():
            np.testing.assert_array_almost_equal(
                expected_df[col],
                calculated_df[col],
            )
    cleanup()
