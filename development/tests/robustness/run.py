#!/usr/bin/env python
"""This script checks whether the package performs properly for random requests."""
import datetime
import sys

from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init
from soepy.soepy_config import TEST_RESOURCES_DIR


def func(maxrt):
    stop = datetime.datetime.now() + maxrt
    while datetime.datetime.now() < stop:
        model_spec_init_dict, random_model_params_df = random_init()

        model_spec_init_dict["EXOG_PROC"]["kids_info_file_name"] = (
            str(TEST_RESOURCES_DIR) + "/" + "exog_child_info.pkl"
        )
        model_spec_init_dict["EXOG_PROC"]["educ_info_file_name"] = (
            str(TEST_RESOURCES_DIR) + "/" + "exog_educ_info_generic.pkl"
        )

        simulate(random_model_params_df, model_spec_init_dict)


if __name__ == "__main__":

    minutes = float(sys.argv[1])
    func(datetime.timedelta(minutes=0.1))
