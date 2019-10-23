#!/usr/bin/env python
"""This module contains the process that generates our regression test battery."""
import argparse
import pickle
import numpy as np

from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.test.random_init import random_init
from development.tests.auxiliary.auxiliary import cleanup


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


def create_vault(num_test=1000, seed=123456):
    """This function creates our regression vault."""
    np.random.seed(seed)
    seeds = np.random.randint(0, 1000, size=num_test)
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    tests = []

    for counter, seed in enumerate(seeds):

        np.random.seed(seed)

        model_spec_init_dict, random_model_params_df = random_init()

        df = simulate("test.soepy.pkl", "test.soepy.yml")

        tests += [(model_spec_init_dict, random_model_params_df, df)]

    cleanup("regression")

    with open(vault, "wb") as file:
        pickle.dump(tests, file)


def check_vault(num_test):
    """This function runs another simulation for each init file in our regression vault.
    """
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    for test in tests[:num_test]:

        model_spec_init_dict, random_model_params_df, expected_df = test

        calculated_df = simulate(random_model_params_df, model_spec_init_dict)

        for col in expected_df.columns.tolist():
            expected_df[col].equals(calculated_df[col])

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
        check_vault(num_test)

    elif request == "create":

        create_vault(num_test, seed)
