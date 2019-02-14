""""""
import os
import json

import numpy as np

from soepy.python.simulate.simulate_python import simulate
from soepy.python.soepy_config import TEST_RESOURCES_DIR
from soepy.python.test.random_init import random_init
from soepy.python.test.random_init import print_dict

NUM_TESTS = 100

np.random.seed(1234235)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)
directory = os.path.dirname(__file__)
file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.soepy.json")

# Enable
CHECK = False
tests = []

for counter, seed in enumerate(seeds):

    print(counter)

    np.random.seed(seed)

    init_dict = random_init()

    df = simulate("test.soepy.yml")

    stat = np.sum(df.sum())

    tests += [(stat, init_dict)]

json.dump(tests, open(file_dir, "w"))

if CHECK:
    tests = json.load(open(file_dir, "r"))
    for test in tests:

        stat, init_dict = test

        print_dict(init_dict)

        df = simulate("test.grmpy.yml")

        stat_new = np.sum(df.sum())

        np.testing.assert_array_almost_equal(stat, stat_new)
