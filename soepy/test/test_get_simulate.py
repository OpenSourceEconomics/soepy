import numpy as np
import pandas as pd

from soepy.simulate.simulate_python import get_simulate_func
from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.resources.aux_funcs import cleanup
from soepy.test.resources.initial_states import create_initial_states


def test_simulation_func():
    """Check that simulate() and get_simulate_func() agree.

    This is an API-consistency test; it does not validate levels against regression
    targets. It uses `random_init` to generate inputs.
    """

    constr = {
        "AGENTS": 200,
        "PERIODS": 6,
        "EDUC_YEARS": [0, 1, 3],
        "CHILD_AGE_INIT_MAX": 1,
        "INIT_EXP_MAX": 1,
        "SEED_SIM": 1234,
        "SEED_EMAX": 4321,
        "NUM_DRAWS_EMAX": 30,
    }
    random_init(constr)

    initial_states = create_initial_states(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
    )

    df_sim = simulate(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
        initial_states=initial_states,
    )
    simulate_func = get_simulate_func(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
        initial_states=initial_states,
    )
    df_partial_sim = simulate_func(
        model_params_init_file_name_inner="test.soepy.pkl",
        model_spec_init_file_name_inner="test.soepy.yml",
    )

    pd.testing.assert_series_equal(
        df_sim.sum(axis=0),
        df_partial_sim.sum(axis=0),
    )

    # Bellman consistency check at period 0: value functions must equal
    # flow utility plus discounted continuation value.
    df0 = df_sim.reset_index().loc[lambda x: x["Period"] == 0]

    params_df = pd.read_pickle("test.soepy.pkl")
    delta = float(params_df.loc[("discount", "delta"), "value"])

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
    for suffix in ["N", "P", "F"]:
        np.testing.assert_array_equal(
            df_sim[f"Taste_Shock_{suffix}"].to_numpy(),
            np.zeros(df_sim.shape[0]),
        )

    cleanup()


def test_simulation_choices_include_taste_shocks():
    constr = {
        "AGENTS": 100,
        "PERIODS": 4,
        "EDUC_YEARS": [0, 1, 2],
        "CHILD_AGE_INIT_MAX": 1,
        "INIT_EXP_MAX": 1,
        "SEED_SIM": 4321,
        "SEED_EMAX": 1234,
        "NUM_DRAWS_EMAX": 20,
    }
    random_init(constr)

    params_df = pd.read_pickle("test.soepy.pkl")
    params_df.loc[("taste_shock", "lambda_taste"), "value"] = 0.5

    initial_states = create_initial_states(
        model_params_init_file_name=params_df,
        model_spec_init_file_name="test.soepy.yml",
    )

    df_sim = simulate(
        model_params_init_file_name=params_df,
        model_spec_init_file_name="test.soepy.yml",
        initial_states=initial_states,
    )

    value_functions = df_sim[
        ["Value_Function_N", "Value_Function_P", "Value_Function_F"]
    ].to_numpy()
    taste_shocks = df_sim[
        ["Taste_Shock_N", "Taste_Shock_P", "Taste_Shock_F"]
    ].to_numpy()

    assert not np.allclose(taste_shocks, 0.0)
    np.testing.assert_array_equal(
        df_sim["Choice"].to_numpy(),
        np.argmax(value_functions + taste_shocks, axis=1),
    )

    cleanup()
