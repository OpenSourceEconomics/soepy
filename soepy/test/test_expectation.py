import pickle
import random

import pandas as pd
import pytest

from development.tests.auxiliary.auxiliary import cleanup
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR


CASES_TEST = random.sample(range(0, 100), 10)


@pytest.fixture(scope="module")
def input_vault():
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    return tests


@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func_exp(input_vault, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
    (
        model_spec_init_dict,
        random_model_params_df,
        exog_educ_shares,
        exog_child_age_shares,
        exog_partner_shares,
        exog_exper_shares_pt,
        exog_exper_shares_ft,
        exog_child_info,
        exog_partner_arrival_info,
        exog_partner_separation_info,
        expected_df,
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")
    #

    calculated_df_false = simulate(
        random_model_params_df, model_spec_init_dict, is_expected=False
    )

    for edu_type in ["low", "middle", "high"]:
        random_model_params_df.loc[
            ("exp_returns_p_bias", f"gamma_p_bias_{edu_type}"), "value"
        ] = (
            random_model_params_df.loc[
                ("exp_returns_p", f"gamma_p_{edu_type}"), "value"
            ]
            / random_model_params_df.loc[
                ("exp_returns_f", f"gamma_f_{edu_type}"), "value"
            ]
        )

    calculated_df_true = simulate(
        random_model_params_df, model_spec_init_dict, is_expected=True
    )

    pd.testing.assert_series_equal(
        calculated_df_false.sum(axis=0),
        calculated_df_true.sum(axis=0),
    )
    cleanup()
