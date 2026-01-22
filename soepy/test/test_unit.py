import numpy as np

from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init


def test_unit_nan():
    """Education starts after completion.

    For each education group, the first observed period in the simulated sample must be
    the number of education years for that group.
    """

    constr = {
        "AGENTS": 200,
        "PERIODS": 7,
        "EDUC_YEARS": [0, 2, 4],
        "INIT_EXP_MAX": 0,
        "CHILD_AGE_INIT_MAX": -1,
        "SEED_SIM": 1234,
        "SEED_EMAX": 4321,
        "NUM_DRAWS_EMAX": 20,
    }
    random_init(constr)
    df = simulate(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
    ).reset_index()

    for educ_level, educ_years in enumerate(constr["EDUC_YEARS"]):
        first_period = df[df["Education_Level"] == educ_level]["Period"].min()
        np.testing.assert_equal(first_period, educ_years)


def test_no_children_no_exp():
    """If child birth probs are zero and init exp is constrained to 0.

    Then:
    - child age stays -1 (no child) in the simulated sample
    - initial part-time and full-time experience years are zero
    """

    constr = {
        "AGENTS": 200,
        "PERIODS": 6,
        "CHILD_AGE_INIT_MAX": -1,
        "INIT_EXP_MAX": 0,
        "SEED_SIM": 1111,
        "SEED_EMAX": 2222,
        "NUM_DRAWS_EMAX": 20,
    }
    random_init(constr)

    df = simulate(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
    ).reset_index()

    # If the initial child-age distribution is degenerate at -1, then at period 0 all
    # agents must have no child.
    df0 = df[df["Period"] == 0].dropna()
    np.testing.assert_equal(int((df0["Age_Youngest_Child"] != -1).sum()), 0)

    np.testing.assert_equal(int((df0["Experience_Part_Time"] != 0).sum()), 0)
    np.testing.assert_equal(int((df0["Experience_Full_Time"] != 0).sum()), 0)


def test_unit_data_frame_shape():
    """Simulated rows match expected entry timing.

    The simulation only includes individuals after they have finished education.
    Therefore, for each education group e, its agents contribute only
    (PERIODS - EDUC_YEARS[e]) rows.
    """

    constr = {
        "AGENTS": 120,
        "PERIODS": 8,
        "EDUC_YEARS": [0, 1, 3],
        "INIT_EXP_MAX": 0,
        "CHILD_AGE_INIT_MAX": -1,
        "SEED_SIM": 3333,
        "SEED_EMAX": 4444,
        "NUM_DRAWS_EMAX": 20,
    }
    random_init(constr)

    df = simulate(
        model_params_init_file_name="test.soepy.pkl",
        model_spec_init_file_name="test.soepy.yml",
    ).reset_index()

    counts = [df[df["Education_Level"] == i]["Identifier"].nunique() for i in [0, 1, 2]]

    expected_rows = sum(
        n_agents * (constr["PERIODS"] - edu_years)
        for n_agents, edu_years in zip(counts, constr["EDUC_YEARS"], strict=True)
    )
    np.testing.assert_equal(df.shape[0], expected_rows)
