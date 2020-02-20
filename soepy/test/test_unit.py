import collections

import numpy as np
from random import randrange, randint

from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.solve.solve_auxiliary import pyth_create_state_space
from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.random_init import read_init_file2
from soepy.test.random_init import namedtuple_to_dict
from soepy.test.random_init import init_dict_flat_to_init_dict
from soepy.solve.solve_python import pyth_solve
from soepy.simulate.simulate_auxiliary import pyth_simulate


def test_unit_nan():
    """This test ensures that the data frame contain only nan values if individuals are
     still a in education.
    """
    constr = {"AGENTS": 200}
    random_init(constr)
    df = simulate("test.soepy.pkl", "test.soepy.yml")

    for year in [11, 12]:

        df2 = df[(df["Years_of_Education"] == year) & (df["Period"] < year - 10)]

        df2 = df2[
            [
                col
                for col in df2.columns.values
                if col not in ["Identifier", "Period", "Years_of_Education"]
            ]
        ]
        a = np.empty(df2.shape)
        a[:] = np.nan

        np.testing.assert_array_equal(df2.values, a)


def test_unit_init_print():
    """This test ensures that the init file printing process work as intended. For this
     purpose we generate random init file specifications import the resulting files,
     write the specifications to another init file, import it again and comparing both
      initialization dicts
      """
    order = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "EXOG_PROC",
    ]

    for _ in range(5):
        random_init()
        model_params_df, _ = read_model_params_init("test.soepy.pkl")
        model_spec = read_model_spec_init("test.soepy.yml", model_params_df)
        init_dict_flat = namedtuple_to_dict(model_spec)
        init_dict = init_dict_flat_to_init_dict(init_dict_flat)
        init_dict2 = read_init_file2("test.soepy.yml")

        for key in order:
            for subkey in init_dict[key].keys():
                if not init_dict[key][subkey] == init_dict2[key][subkey]:
                    raise AssertionError()


def test_unit_data_frame_shape():
    """This test ensures that the shape of the simulated data frame corresponds to the
    to the random specifications of our initialization file.
    """
    for _ in range(5):
        constr = dict()
        constr["AGENTS"] = np.random.randint(10, 100)
        constr["PERIODS"] = np.random.randint(1, 6)
        constr["EDUC_MAX"] = np.random.randint(10, min(10 + constr["PERIODS"], 12))

        random_init(constr)
        df = simulate("test.soepy.pkl", "test.soepy.yml")

        np.testing.assert_array_equal(df.shape[0], constr["AGENTS"] * constr["PERIODS"])


def test_unit_states_hard_code():
    """This test ensures that the state space creation generates the correct admissible
    state space points for the first 4 periods."""

    model_spec = collections.namedtuple(
        "model_spec",
        "num_periods educ_range educ_min num_types \
         last_child_bearing_period, child_age_max",
    )
    model_spec = model_spec(3, 3, 10, 2, 24, 12)

    states, _ = pyth_create_state_space(model_spec)

    states_true = [
        [0, 10, 0, 0, 0, 0, -1],
        [0, 10, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 1, -1],
        [0, 10, 0, 0, 0, 1, 0],
        [1, 10, 0, 0, 0, 0, -1],
        [1, 10, 1, 1, 0, 0, -1],
        [1, 10, 2, 0, 1, 0, -1],
        [1, 11, 0, 0, 0, 0, -1],
        [1, 10, 0, 0, 0, 0, 0],
        [1, 10, 1, 1, 0, 0, 0],
        [1, 10, 2, 0, 1, 0, 0],
        [1, 11, 0, 0, 0, 0, 0],
        [1, 10, 0, 0, 0, 0, 1],
        [1, 10, 1, 1, 0, 0, 1],
        [1, 10, 2, 0, 1, 0, 1],
        [1, 11, 0, 0, 0, 0, 1],
        [1, 10, 0, 0, 0, 1, -1],
        [1, 10, 1, 1, 0, 1, -1],
        [1, 10, 2, 0, 1, 1, -1],
        [1, 11, 0, 0, 0, 1, -1],
        [1, 10, 0, 0, 0, 1, 0],
        [1, 10, 1, 1, 0, 1, 0],
        [1, 10, 2, 0, 1, 1, 0],
        [1, 11, 0, 0, 0, 1, 0],
        [1, 10, 0, 0, 0, 1, 1],
        [1, 10, 1, 1, 0, 1, 1],
        [1, 10, 2, 0, 1, 1, 1],
        [1, 11, 0, 0, 0, 1, 1],
        [2, 10, 0, 0, 0, 0, -1],
        [2, 10, 0, 1, 0, 0, -1],
        [2, 10, 1, 1, 0, 0, -1],
        [2, 10, 1, 2, 0, 0, -1],
        [2, 10, 0, 0, 1, 0, -1],
        [2, 10, 2, 0, 1, 0, -1],
        [2, 10, 1, 1, 1, 0, -1],
        [2, 10, 2, 1, 1, 0, -1],
        [2, 10, 2, 0, 2, 0, -1],
        [2, 11, 0, 0, 0, 0, -1],
        [2, 11, 1, 1, 0, 0, -1],
        [2, 11, 2, 0, 1, 0, -1],
        [2, 12, 0, 0, 0, 0, -1],
        [2, 10, 0, 0, 0, 0, 0],
        [2, 10, 0, 1, 0, 0, 0],
        [2, 10, 1, 1, 0, 0, 0],
        [2, 10, 1, 2, 0, 0, 0],
        [2, 10, 0, 0, 1, 0, 0],
        [2, 10, 2, 0, 1, 0, 0],
        [2, 10, 1, 1, 1, 0, 0],
        [2, 10, 2, 1, 1, 0, 0],
        [2, 10, 2, 0, 2, 0, 0],
        [2, 11, 0, 0, 0, 0, 0],
        [2, 11, 1, 1, 0, 0, 0],
        [2, 11, 2, 0, 1, 0, 0],
        [2, 12, 0, 0, 0, 0, 0],
        [2, 10, 0, 0, 0, 0, 1],
        [2, 10, 0, 1, 0, 0, 1],
        [2, 10, 1, 1, 0, 0, 1],
        [2, 10, 1, 2, 0, 0, 1],
        [2, 10, 0, 0, 1, 0, 1],
        [2, 10, 2, 0, 1, 0, 1],
        [2, 10, 1, 1, 1, 0, 1],
        [2, 10, 2, 1, 1, 0, 1],
        [2, 10, 2, 0, 2, 0, 1],
        [2, 11, 0, 0, 0, 0, 1],
        [2, 11, 1, 1, 0, 0, 1],
        [2, 11, 2, 0, 1, 0, 1],
        [2, 12, 0, 0, 0, 0, 1],
        [2, 10, 0, 0, 0, 0, 2],
        [2, 10, 0, 1, 0, 0, 2],
        [2, 10, 1, 1, 0, 0, 2],
        [2, 10, 1, 2, 0, 0, 2],
        [2, 10, 0, 0, 1, 0, 2],
        [2, 10, 2, 0, 1, 0, 2],
        [2, 10, 1, 1, 1, 0, 2],
        [2, 10, 2, 1, 1, 0, 2],
        [2, 10, 2, 0, 2, 0, 2],
        [2, 11, 0, 0, 0, 0, 2],
        [2, 11, 1, 1, 0, 0, 2],
        [2, 11, 2, 0, 1, 0, 2],
        [2, 12, 0, 0, 0, 0, 2],
        [2, 10, 0, 0, 0, 1, -1],
        [2, 10, 0, 1, 0, 1, -1],
        [2, 10, 1, 1, 0, 1, -1],
        [2, 10, 1, 2, 0, 1, -1],
        [2, 10, 0, 0, 1, 1, -1],
        [2, 10, 2, 0, 1, 1, -1],
        [2, 10, 1, 1, 1, 1, -1],
        [2, 10, 2, 1, 1, 1, -1],
        [2, 10, 2, 0, 2, 1, -1],
        [2, 11, 0, 0, 0, 1, -1],
        [2, 11, 1, 1, 0, 1, -1],
        [2, 11, 2, 0, 1, 1, -1],
        [2, 12, 0, 0, 0, 1, -1],
        [2, 10, 0, 0, 0, 1, 0],
        [2, 10, 0, 1, 0, 1, 0],
        [2, 10, 1, 1, 0, 1, 0],
        [2, 10, 1, 2, 0, 1, 0],
        [2, 10, 0, 0, 1, 1, 0],
        [2, 10, 2, 0, 1, 1, 0],
        [2, 10, 1, 1, 1, 1, 0],
        [2, 10, 2, 1, 1, 1, 0],
        [2, 10, 2, 0, 2, 1, 0],
        [2, 11, 0, 0, 0, 1, 0],
        [2, 11, 1, 1, 0, 1, 0],
        [2, 11, 2, 0, 1, 1, 0],
        [2, 12, 0, 0, 0, 1, 0],
        [2, 10, 0, 0, 0, 1, 1],
        [2, 10, 0, 1, 0, 1, 1],
        [2, 10, 1, 1, 0, 1, 1],
        [2, 10, 1, 2, 0, 1, 1],
        [2, 10, 0, 0, 1, 1, 1],
        [2, 10, 2, 0, 1, 1, 1],
        [2, 10, 1, 1, 1, 1, 1],
        [2, 10, 2, 1, 1, 1, 1],
        [2, 10, 2, 0, 2, 1, 1],
        [2, 11, 0, 0, 0, 1, 1],
        [2, 11, 1, 1, 0, 1, 1],
        [2, 11, 2, 0, 1, 1, 1],
        [2, 12, 0, 0, 0, 1, 1],
        [2, 10, 0, 0, 0, 1, 2],
        [2, 10, 0, 1, 0, 1, 2],
        [2, 10, 1, 1, 0, 1, 2],
        [2, 10, 1, 2, 0, 1, 2],
        [2, 10, 0, 0, 1, 1, 2],
        [2, 10, 2, 0, 1, 1, 2],
        [2, 10, 1, 1, 1, 1, 2],
        [2, 10, 2, 1, 1, 1, 2],
        [2, 10, 2, 0, 2, 1, 2],
        [2, 11, 0, 0, 0, 1, 2],
        [2, 11, 1, 1, 0, 1, 2],
        [2, 11, 2, 0, 1, 1, 2],
        [2, 12, 0, 0, 0, 1, 2],
    ]

    np.testing.assert_array_equal(states_true, states)


def test_unit_childbearing_age():
    """This test verifies that the state space does not contain newly born children
    after the last childbearing period"""
    expected = 0

    model_spec = collections.namedtuple(
        "model_spec",
        "num_periods educ_range educ_min num_types \
        last_child_bearing_period child_age_max",
    )

    num_periods = randint(1, 11)
    last_child_bearing_period = randrange(num_periods)
    model_spec = model_spec(num_periods, 3, 10, 2, last_child_bearing_period, 12)

    states, _ = pyth_create_state_space(model_spec)

    np.testing.assert_equal(
        sum(
            states[np.where(states[:, 0] == model_spec.last_child_bearing_period + 1)][
                :, 6
            ]
            == 0
        ),
        expected,
    )


def test_no_children_prob_0():
    """This test ensures that child age equals -1 in the entire simulates sample,
    equivalent to no kid is ever born, if the probability to get a child is zero
    for all periods"""

    expected = 0

    is_expected = False

    constr = {"AGENTS": 200, "PERIODS": 6}
    random_init(constr)

    model_params_df, model_params = read_model_params_init("test.soepy.pkl")
    model_spec = read_model_spec_init("test.soepy.yml", model_params_df)

    # Set probability of having children to zero for all periods
    prob_child = np.full(model_spec.num_periods, 0.00)

    # Solve
    states, indexer, covariates, emaxs, child_age_update_rule = pyth_solve(
        model_params, model_spec, prob_child, is_expected
    )

    # Simulate
    df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
        child_age_update_rule,
        prob_child,
        is_expected=False,
    )

    np.testing.assert_equal(sum(df.dropna()["Age_Youngest_Child"] != -1), expected)
