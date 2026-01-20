import pickle
import random

import jax.numpy as jnp
import pandas as pd
import pytest

from soepy.simulate.simulate_python import get_simulate_func
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.test.resources.aux_funcs import cleanup

CASES_TEST = random.sample(range(0, 100), 10)


@pytest.fixture(scope="module")
def input_vault():
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    return tests


@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func(input_vault, test_id):
    """Check that simulate() and get_simulate_func() agree.

    This is an API-consistency test; it does not validate levels against regression
    targets.
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
        expected_df_unbiased,
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    # Add new inputs for continuous experience model
    exp_grid = jnp.linspace(0.0, 1.0, 10)
    model_spec_init_dict["exp_grid"] = exp_grid

    for old_category, new_category, old_param_name, new_param_name in [
        ("exp_returns_f", "exp_return", "gamma_f", "gamma_1"),
        ("exp_returns_p", "exp_increase_p", "gamma_p", "gamma_p"),
        ("exp_returns_p_bias", "exp_increase_p_bias", "gamma_p_bias", "gamma_p_bias"),
    ]:
        for educ_ind, educ_type in enumerate(["low", "middle", "high"]):
            random_model_params_df.loc[
                (new_category, f"{new_param_name}_{educ_type}"), "value"
            ] = random_model_params_df.loc[
                (old_category, f"{old_param_name}_{educ_type}"), "value"
            ]
            # Delete old entry
            random_model_params_df = random_model_params_df.drop(
                index=(old_category, f"{old_param_name}_{educ_type}")
            )

    # Sort index after modifications
    random_model_params_df = random_model_params_df.sort_index()

    df_sim = simulate(random_model_params_df, model_spec_init_dict)
    simulate_func = get_simulate_func(random_model_params_df, model_spec_init_dict)
    df_partial_sim = simulate_func(random_model_params_df, model_spec_init_dict)

    pd.testing.assert_series_equal(
        df_sim.sum(axis=0),
        df_partial_sim.sum(axis=0),
    )
    cleanup()
