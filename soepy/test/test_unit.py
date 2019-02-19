import numpy as np

from soepy.python.shared.shared_auxiliary import calculate_consumption_utilities
from soepy.python.shared.shared_auxiliary import calculate_total_utilities
from soepy.python.shared.shared_auxiliary import calculate_wage_systematic
from soepy.python.shared.shared_auxiliary import calculate_period_wages
from soepy.python.pre_processing.model_processing import read_init_file
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.test.random_init import print_dict
from soepy.test.auxiliary import cleanup


def test1():
    """This test ensures that the columns of the output dataframe correspond to the
    to the .
    """
    for _ in range(100):
        constr = {"EDUC_MAX": 10, "AGENTS": 1, "PERIODS": 1}
        init_dict = random_init(constr)
        attr_dict = read_init_file("test.soepy.yml")
        df = simulate("test.soepy.yml")

        optim_paras = init_dict["PARAMETERS"]["optim_paras"]

        educ_level = np.array([1.0, 0.0, 0.0])

        exp_p, exp_f = 0.0, 0.0

        wage_systematic = calculate_wage_systematic(
            educ_level, exp_p, exp_f, optim_paras
        )

        np.testing.assert_array_equal(wage_systematic, df["Systematic Wage"])
        draw_sim = draw_disturbances(
            (1, 1),
            attr_dict["DERIVED_ATTR"]["shocks_cov"],
            init_dict["SIMULATION"]["seed_sim"],
        )
        period_wages = calculate_period_wages(
            attr_dict, wage_systematic, draw_sim[0, 0, :]
        )
        period_wages
        np.testing.assert_array_equal(
            period_wages,
            np.squeeze(
                df[["Period Wage N", "Period Wage P", "Period Wage F"]].values.T
            ),
        )

        consumption_utilities = calculate_consumption_utilities(attr_dict, period_wages)

        np.testing.assert_array_equal(
            consumption_utilities,
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

        total_utilities = calculate_total_utilities(
            attr_dict, consumption_utilities, optim_paras
        )

        np.testing.assert_array_equal(
            total_utilities,
            np.squeeze(
                df[["Flow Utility N", "Flow Utility P", "Flow Utility F"]].values
            ),
        )


def test2():
    """This test ensures that the dataframe contain only nan values if individuals are
     still a in education.
    """
    constr = {"AGENTS": 200}
    random_init(constr)
    df = simulate("test.soepy.yml")

    for year in [11, 12, 13, 14]:

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
        init_dict = read_init_file("test.soepy.yml")
        print_dict(init_dict)
        init_dict2 = read_init_file("test.soepy.yml")

        for key in order:
            for subkey in init_dict[key].keys():
                if not init_dict[key][subkey] == init_dict2[key][subkey]:
                    raise AssertionError()


def test4():
    """This test ensures that the shape of the simulated dataframe corresponds to the
    to the random specifications of our initialization file.
    """
    for _ in range(5):
        constr = dict()
        constr["AGENTS"] = np.random.randint(10, 100)
        constr["PERIODS"] = np.random.randint(1, 5)
        constr["EDUC_MAX"] = np.random.randint(10, 10 + constr["PERIODS"])

        random_init(constr)
        df = simulate("test.soepy.yml")

        np.testing.assert_array_equal(df.shape[0], constr["AGENTS"] * constr["PERIODS"])


cleanup()
