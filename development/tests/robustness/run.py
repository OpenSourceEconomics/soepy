#!/usr/bin/env python
"""This script checks whether the package performs properly for random requests."""
import datetime
import sys

from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from development.tests.auxiliary.auxiliary import cleanup


def func(maxrt):
    stop = datetime.datetime.now() + maxrt
    while datetime.datetime.now() < stop:
        (
            model_spec_init_dict,
            random_model_params_df,
            exog_child_info,
            exog_educ_info,
        ) = random_init()

        simulate(random_model_params_df, model_spec_init_dict)


if __name__ == "__main__":

    minutes = float(sys.argv[1])
    func(datetime.timedelta(minutes=0.1))

cleanup()
