import pandas as pd

from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.resources.aux_funcs import cleanup


def test_simulation_func_exp():
    """Expected vs non-expected coincide if pt increment is the same.

    This test relies only on inputs, so we generate them via `random_init`.
    """

    constr = {
        "AGENTS": 200,
        "PERIODS": 6,
        "EDUC_YEARS": [0, 1, 3],
        "CHILD_AGE_INIT_MAX": 1,
        "INIT_EXP_MAX": 1,
        "SEED_SIM": 2024,
        "SEED_EMAX": 2025,
        "NUM_DRAWS_EMAX": 30,
    }
    random_init(constr)

    model_params_df = pd.read_pickle("test.soepy.pkl")

    calculated_df_false = simulate(
        model_params_init_file_name=model_params_df,
        model_spec_init_file_name="test.soepy.yml",
        is_expected=False,
    )

    # Force expected law of motion to match unbiased.
    for edu_type in ["low", "middle", "high"]:
        model_params_df.loc[
            ("exp_increase_p_bias", f"gamma_p_bias_{edu_type}"), "value"
        ] = model_params_df.loc[("exp_increase_p", f"gamma_p_{edu_type}"), "value"]

    calculated_df_true = simulate(
        model_params_init_file_name=model_params_df,
        model_spec_init_file_name="test.soepy.yml",
        is_expected=True,
    )

    pd.testing.assert_series_equal(
        calculated_df_false.sum(axis=0),
        calculated_df_true.sum(axis=0),
    )
    cleanup()
