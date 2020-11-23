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
            exog_educ_shares,
            exog_child_age_shares,
            exog_child_info,
            exog_partner_arrival_info,
        ) = random_init()

        model_spec_init_dict["EXOG_PROC"][
            "partner_arrival_info_file_name"
        ] = model_spec_init_dict["EXOG_PROC"].pop("partner_info_file_name")

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.pkl")

        simulate(random_model_params_df, model_spec_init_dict)


if __name__ == "__main__":

    minutes = float(sys.argv[1])
    func(datetime.timedelta(minutes=0.1))

cleanup()
