""
import json

import numpy as np

from soepy.python.simulate.simulate_python import simulate
from soepy.python.soepy_config import TEST_RESOURCES_DIR
from soepy.test.random_init import print_dict
from soepy.test.auxiliary import cleanup


def test1():
    """This test runs a random selection of five regression tests from the our regression test
    battery.
    """

    fname = TEST_RESOURCES_DIR + "/regression_vault.soepy.json"
    tests = json.load(open(fname))
    random_choice = np.random.choice(range(len(tests)), 3)
    tests = [tests[i] for i in random_choice]

    for test in tests:

        stat, init_dict = test

        print_dict(init_dict)

        df = simulate("test.soepy.yml")

        stat_new = np.sum(df.sum())

        np.testing.assert_array_equal(stat_new, stat)


cleanup()
