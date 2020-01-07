import collections

import numpy as np

from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.solve.solve_auxiliary import pyth_create_state_space
from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.random_init import read_init_file2
from soepy.test.random_init import namedtuple_to_dict
from soepy.test.random_init import init_dict_flat_to_init_dict


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
    order = ["GENERAL", "CONSTANTS", "INITIAL_CONDITIONS", "SIMULATION", "SOLUTION"]

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

    model_params = collections.namedtuple(
        "model_params", "num_periods educ_range educ_min num_types"
    )
    model_params = model_params(4, 3, 10, 2)

    states, _ = pyth_create_state_space(model_params)

    states_true = [
        [0, 10, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 1],
        [1, 10, 0, 0, 0, 0],
        [1, 10, 1, 1, 0, 0],
        [1, 10, 2, 0, 1, 0],
        [1, 11, 0, 0, 0, 0],
        [1, 10, 0, 0, 0, 1],
        [1, 10, 1, 1, 0, 1],
        [1, 10, 2, 0, 1, 1],
        [1, 11, 0, 0, 0, 1],
        [2, 10, 0, 0, 0, 0],
        [2, 10, 0, 1, 0, 0],
        [2, 10, 1, 1, 0, 0],
        [2, 10, 1, 2, 0, 0],
        [2, 10, 0, 0, 1, 0],
        [2, 10, 2, 0, 1, 0],
        [2, 10, 1, 1, 1, 0],
        [2, 10, 2, 1, 1, 0],
        [2, 10, 2, 0, 2, 0],
        [2, 11, 0, 0, 0, 0],
        [2, 11, 1, 1, 0, 0],
        [2, 11, 2, 0, 1, 0],
        [2, 12, 0, 0, 0, 0],
        [2, 10, 0, 0, 0, 1],
        [2, 10, 0, 1, 0, 1],
        [2, 10, 1, 1, 0, 1],
        [2, 10, 1, 2, 0, 1],
        [2, 10, 0, 0, 1, 1],
        [2, 10, 2, 0, 1, 1],
        [2, 10, 1, 1, 1, 1],
        [2, 10, 2, 1, 1, 1],
        [2, 10, 2, 0, 2, 1],
        [2, 11, 0, 0, 0, 1],
        [2, 11, 1, 1, 0, 1],
        [2, 11, 2, 0, 1, 1],
        [2, 12, 0, 0, 0, 1],
        [3, 10, 0, 0, 0, 0],
        [3, 10, 0, 1, 0, 0],
        [3, 10, 1, 1, 0, 0],
        [3, 10, 0, 2, 0, 0],
        [3, 10, 1, 2, 0, 0],
        [3, 10, 1, 3, 0, 0],
        [3, 10, 0, 0, 1, 0],
        [3, 10, 2, 0, 1, 0],
        [3, 10, 0, 1, 1, 0],
        [3, 10, 1, 1, 1, 0],
        [3, 10, 2, 1, 1, 0],
        [3, 10, 1, 2, 1, 0],
        [3, 10, 2, 2, 1, 0],
        [3, 10, 0, 0, 2, 0],
        [3, 10, 2, 0, 2, 0],
        [3, 10, 1, 1, 2, 0],
        [3, 10, 2, 1, 2, 0],
        [3, 10, 2, 0, 3, 0],
        [3, 11, 0, 0, 0, 0],
        [3, 11, 0, 1, 0, 0],
        [3, 11, 1, 1, 0, 0],
        [3, 11, 1, 2, 0, 0],
        [3, 11, 0, 0, 1, 0],
        [3, 11, 2, 0, 1, 0],
        [3, 11, 1, 1, 1, 0],
        [3, 11, 2, 1, 1, 0],
        [3, 11, 2, 0, 2, 0],
        [3, 12, 0, 0, 0, 0],
        [3, 12, 1, 1, 0, 0],
        [3, 12, 2, 0, 1, 0],
        [3, 10, 0, 0, 0, 1],
        [3, 10, 0, 1, 0, 1],
        [3, 10, 1, 1, 0, 1],
        [3, 10, 0, 2, 0, 1],
        [3, 10, 1, 2, 0, 1],
        [3, 10, 1, 3, 0, 1],
        [3, 10, 0, 0, 1, 1],
        [3, 10, 2, 0, 1, 1],
        [3, 10, 0, 1, 1, 1],
        [3, 10, 1, 1, 1, 1],
        [3, 10, 2, 1, 1, 1],
        [3, 10, 1, 2, 1, 1],
        [3, 10, 2, 2, 1, 1],
        [3, 10, 0, 0, 2, 1],
        [3, 10, 2, 0, 2, 1],
        [3, 10, 1, 1, 2, 1],
        [3, 10, 2, 1, 2, 1],
        [3, 10, 2, 0, 3, 1],
        [3, 11, 0, 0, 0, 1],
        [3, 11, 0, 1, 0, 1],
        [3, 11, 1, 1, 0, 1],
        [3, 11, 1, 2, 0, 1],
        [3, 11, 0, 0, 1, 1],
        [3, 11, 2, 0, 1, 1],
        [3, 11, 1, 1, 1, 1],
        [3, 11, 2, 1, 1, 1],
        [3, 11, 2, 0, 2, 1],
        [3, 12, 0, 0, 0, 1],
        [3, 12, 1, 1, 0, 1],
        [3, 12, 2, 0, 1, 1],
    ]

    np.testing.assert_array_equal(states_true, states[0:96, :])
