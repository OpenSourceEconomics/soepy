from collections import namedtuple

import numpy as np

from soepy.python.shared.shared_auxiliary import calculate_consumption_utilities
from soepy.python.shared.shared_auxiliary import calculate_total_utilities
from soepy.python.shared.shared_auxiliary import calculate_wage_systematic
from soepy.python.shared.shared_auxiliary import calculate_period_wages
from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import construct_covariates
from soepy.python.pre_processing.model_processing import read_init_file
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_constants import MISSING_INT
from soepy.python.simulate.simulate_python import simulate
from soepy.python.shared.shared_helpers import convert_state_space
from soepy.test.random_init import random_init
from soepy.test.random_init import read_init_file2
from soepy.test.random_init import namedtuple_to_dict
from soepy.test.random_init import init_dict_flat_to_init_dict
from soepy.test.auxiliary import cleanup


def test1():
    """This test ensures that the columns of the output data frame correspond to the
    function output values.
    """
    for _ in range(100):
        constr = {"EDUC_MAX": 10, "AGENTS": 1, "PERIODS": 1}
        random_init(constr)
        model_params = read_init_file("test.soepy.yml")
        df = simulate("test.soepy.yml")

        states, _ = pyth_create_state_space(model_params)
        state_space_args = convert_state_space(model_params, states)
        covariates = construct_covariates(state_space_args)

        # Test systematic wages
        wage_systematic = calculate_wage_systematic(
            model_params, state_space_args, covariates
        )

        np.testing.assert_array_equal(wage_systematic[0, 0], df["Systematic Wage"])

        # Test period wages
        draw_sim = draw_disturbances(
            model_params.seed_sim, model_params.shocks_cov, 1, 1
        )
        period_wages = calculate_period_wages(
            model_params, state_space_args, wage_systematic, draw_sim
        )

        np.testing.assert_array_equal(
            period_wages[0, 0, 0, :],
            np.squeeze(
                df[["Period Wage N", "Period Wage P", "Period Wage F"]].values.T
            ),
        )

        # Test consumption utilities
        consumption_utilities = calculate_consumption_utilities(
            model_params, period_wages
        )

        np.testing.assert_array_equal(
            consumption_utilities[0, 0, 0, :],
            np.squeeze(
                df[
                    [
                        "Consumption Utility N",
                        "Consumption Utility P",
                        "Consumption Utility F",
                    ]
                ].values.T
            ),
        )

        # Test total utilities
        flow_utilities = calculate_total_utilities(model_params, consumption_utilities)

        np.testing.assert_array_equal(
            flow_utilities[0, 0, 0, :],
            np.squeeze(
                df[["Flow Utility N", "Flow Utility P", "Flow Utility F"]].values
            ),
        )


def test2():
    """This test ensures that the data frame contain only nan values if individuals are
     still a in education.
    """
    constr = {"AGENTS": 200}
    random_init(constr)
    df = simulate("test.soepy.yml")

    for year in [11, 12]:

        df2 = df[(df["Years of Education"] == year) & (df["Period"] < year - 10)]

        df2 = df2[
            [
                col
                for col in df2.columns.values
                if col not in ["Identifier", "Period", "Years of Education"]
            ]
        ]
        a = np.empty(df2.shape)
        a[:] = np.nan

        np.testing.assert_array_equal(df2.values, a)


def test3():
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
        "PARAMETERS",
    ]

    for _ in range(5):
        random_init()
        model_params = read_init_file("test.soepy.yml")
        init_dict_flat = namedtuple_to_dict(model_params)
        init_dict = init_dict_flat_to_init_dict(init_dict_flat)
        init_dict2 = read_init_file2("test.soepy.yml")

        for key in order:
            for subkey in init_dict[key].keys():
                if not init_dict[key][subkey] == init_dict2[key][subkey]:
                    raise AssertionError()


def test4():
    """This test ensures that the shape of the simulated data frame corresponds to the
    to the random specifications of our initialization file.
    """
    for _ in range(5):
        constr = dict()
        constr["AGENTS"] = np.random.randint(10, 100)
        constr["PERIODS"] = np.random.randint(1, 6)
        constr["EDUC_MAX"] = np.random.randint(10, min(10 + constr["PERIODS"], 12))

        random_init(constr)
        df = simulate("test.soepy.yml")

        np.testing.assert_array_equal(df.shape[0], constr["AGENTS"] * constr["PERIODS"])


def test5():
    """This test ensures that the state space creation generates the correct admissible
    state space points for the first 4 periods."""

    model_params = namedtuple("model_params", "num_periods educ_range educ_min")
    model_params = model_params(4, 3, 10)

    states, _ = pyth_create_state_space(model_params)

    states_all, states_number_period, _, _ = convert_state_space(model_params, states)

    # Control for correct number of states in each period.
    np.testing.assert_array_equal(states_number_period, [1, 4, 13, 30])

    # Control that the states are correct
    states_true = np.full((4, 576, 4), MISSING_INT)

    states_true[0, 0, :] = [10, 0, 0, 0]

    states_true[1, 0:4, :] = [
        [10, 0, 0, 0],
        [10, 1, 1, 0],
        [10, 2, 0, 1],
        [11, 0, 0, 0],
    ]

    states_true[2, 0:13, :] = [
        [10, 0, 0, 0],
        [10, 0, 1, 0],
        [10, 1, 1, 0],
        [10, 1, 2, 0],
        [10, 0, 0, 1],
        [10, 2, 0, 1],
        [10, 1, 1, 1],
        [10, 2, 1, 1],
        [10, 2, 0, 2],
        [11, 0, 0, 0],
        [11, 1, 1, 0],
        [11, 2, 0, 1],
        [12, 0, 0, 0],
    ]

    states_true[3, 0:30, :] = [
        [10, 0, 0, 0],
        [10, 0, 1, 0],
        [10, 1, 1, 0],
        [10, 0, 2, 0],
        [10, 1, 2, 0],
        [10, 1, 3, 0],
        [10, 0, 0, 1],
        [10, 2, 0, 1],
        [10, 0, 1, 1],
        [10, 1, 1, 1],
        [10, 2, 1, 1],
        [10, 1, 2, 1],
        [10, 2, 2, 1],
        [10, 0, 0, 2],
        [10, 2, 0, 2],
        [10, 1, 1, 2],
        [10, 2, 1, 2],
        [10, 2, 0, 3],
        [11, 0, 0, 0],
        [11, 0, 1, 0],
        [11, 1, 1, 0],
        [11, 1, 2, 0],
        [11, 0, 0, 1],
        [11, 2, 0, 1],
        [11, 1, 1, 1],
        [11, 2, 1, 1],
        [11, 2, 0, 2],
        [12, 0, 0, 0],
        [12, 1, 1, 0],
        [12, 2, 0, 1],
    ]

    np.testing.assert_array_equal(states_true, states_all)


cleanup()
