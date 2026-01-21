import pickle
import random

import jax.numpy as jnp
import numpy as np
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

    df_sim = simulate(random_model_params_df, model_spec_init_dict)
    simulate_func = get_simulate_func(random_model_params_df, model_spec_init_dict)
    df_partial_sim = simulate_func(random_model_params_df, model_spec_init_dict)

    pd.testing.assert_series_equal(
        df_sim.sum(axis=0),
        df_partial_sim.sum(axis=0),
    )

    # Bellman consistency check at period 0: value functions must equal
    # flow utility plus discounted continuation value.
    delta = float(random_model_params_df.loc[("discount", "delta"), "value"])
    df0 = df_sim.reset_index().loc[lambda x: x["Period"] == 0]

    for suffix in ["N", "P", "F"]:
        np.testing.assert_allclose(
            df0[f"Value_Function_{suffix}"].to_numpy(),
            df0[f"Flow_Utility_{suffix}"].to_numpy()
            + delta * df0[f"Continuation_Value_{suffix}"].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    vf = df0[["Value_Function_N", "Value_Function_P", "Value_Function_F"]].to_numpy()
    np.testing.assert_array_equal(df0["Choice"].to_numpy(), vf.argmax(axis=1))

    cleanup()
