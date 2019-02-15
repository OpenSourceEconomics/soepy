"""This module contains the process that generates our regression test battery."""
import os
import json
import argparse

import numpy as np

from soepy.python.simulate.simulate_python import simulate
from soepy.python.soepy_config import TEST_RESOURCES_DIR
from soepy.test.random_init import random_init
from soepy.test.random_init import print_dict
from soepy.test.auxiliary import cleanup


def process_arguments(parser):
    """This function parses the input arguments."""
    args = parser.parse_args()

    # Distribute input arguments
    request = args.request
    num_test = args.num_test
    seed = args.seed
    # Test validity of input arguments
    assert request in ["check", "create"]

    if num_test is None:
        num_test = 100

    if seed is None:
        seed = 123456

    return request, num_test, seed


def create_vault(num_test=100, seed=123456):
    """This function creates our regression vault."""
    np.random.seed(seed)
    seeds = np.random.randint(0, 1000, size=num_test)
    file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.soepy.json")
    tests = []

    for counter, seed in enumerate(seeds):

        np.random.seed(seed)

        init_dict = random_init()

        df = simulate("test.soepy.yml")

        stat = np.sum(df.sum())

        tests += [(stat, init_dict)]
    cleanup("regression")

    json.dump(tests, open(file_dir, "w"))


def check_vault():
    """This function runs another simulation for each init file in our regression vault.
    """
    file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.soepy.json")

    tests = json.load(open(file_dir, "r"))
    for test in tests:

        stat, init_dict = test

        print_dict(init_dict)

        df = simulate("test.grmpy.yml")

        stat_new = np.sum(df.sum())

        np.testing.assert_array_almost_equal(stat, stat_new)

    cleanup("regression")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Work with regression tests for package."
    )

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        required=True,
        choices=["check", "create"],
        help="request",
    )

    parser.add_argument(
        "--num", action="store", dest="num_test", type=int, help="number of init files"
    )

    parser.add_argument(
        "--seed", action="store", dest="seed", type=int, help="seed value"
    )

    request, num_test, seed = process_arguments(parser)

    if request == "check":
        check_vault()

    elif request == "create":

        create_vault(num_test, seed)
