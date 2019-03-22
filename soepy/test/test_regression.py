""
import json

import numpy as np
from pathlib import Path

from soepy.python.simulate.simulate_python import simulate
from soepy.python.soepy_config import TEST_RESOURCES_DIR

# from soepy.test.random_init import print_dict
from soepy.test.auxiliary import cleanup


# def test1():
#     """This test runs a random selection of five regression tests from
#     our regression test battery.
#     """
#
#     fname = TEST_RESOURCES_DIR / "regression_vault.soepy.json"
#     tests = json.load(open(fname))
#     random_choice = np.random.choice(range(len(tests)), 3)
#     tests = [tests[i] for i in random_choice]
#
#     for test in tests:
#
#         stat, init_dict = test
#
#         print_dict(init_dict)
#
#         df = simulate("test.soepy.yml")
#
#         stat_new = np.sum(df.sum())
#
#         np.testing.assert_array_equal(stat_new, stat)


def test1():
    """This test runs five regression tests using yml ini files.
    """

    pathlist = Path(TEST_RESOURCES_DIR).glob("**/*.yml")
    files = [x for x in pathlist if x.is_file()]

    file_list = []
    for file in files:
        file_list.append(str(file))

    file_list.sort()

    test_fname = TEST_RESOURCES_DIR / "regression_vault.dev.soepy.json"
    tests = json.load(open(test_fname))

    for i in range(len(tests)):
        stat, _, = tests[i]

        df = simulate(file_list[i])

        stat_new = np.sum(df.sum())

        np.testing.assert_array_equal(stat_new, stat)


cleanup()
