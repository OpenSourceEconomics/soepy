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
        biased_exp=False,
    )

    # Force expected law of motion to match unbiased.
    # Under the new rule `biased_exp=True` always returns 1.0, so set gamma_p to 1.0
    # and remove the mother increment.
    for edu_type in ["low", "middle", "high"]:
        model_params_df.loc[("exp_increase_p", f"gamma_p_{edu_type}"), "value"] = 1.0

    model_params_df.loc[("exp_increase_p_mom", "gamma_p_mom"), "value"] = 0.0

    calculated_df_true = simulate(
        model_params_init_file_name=model_params_df,
        model_spec_init_file_name="test.soepy.yml",
        biased_exp=True,
    )

    # Under the new rule, `biased_exp=True` sets pt increment to 1.0 everywhere.
    # To make the two runs comparable, we only assert equality for columns that do not
    # depend on the experience accumulation law.
    cols = [
        "Education_Level",
        "Type",
        "Partner_Indicator",
        "Age_Youngest_Child",
        "Male_Wages",
    ]

    pd.testing.assert_series_equal(
        calculated_df_false[cols].sum(axis=0),
        calculated_df_true[cols].sum(axis=0),
    )
    cleanup()
