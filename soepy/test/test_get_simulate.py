import numpy as np
import pandas as pd

from soepy.simulate.simulate_python import get_simulate_func
from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.resources.aux_funcs import cleanup


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

    df_sim = simulate(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
    )
    simulate_func = get_simulate_func(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
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

    cleanup()
