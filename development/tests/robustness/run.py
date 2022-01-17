#!/usr/bin/env python
"""This script checks whether the package performs properly for random requests."""
import datetime
import sys

from development.tests.auxiliary.auxiliary import cleanup
from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init


def func(maxrt):
    stop = datetime.datetime.now() + maxrt
    while datetime.datetime.now() < stop:
        (
            model_spec_init_dict,
            random_model_params_df,
            exog_educ_shares,
            exog_child_age_shares,
            exog_partner_shares,
            exog_exper_shares_pt,
            exog_exper_shares_ft,
            exog_child_info,
            exog_partner_arrival_info,
            exog_partner_separation_info,
        ) = random_init()

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

        simulate(random_model_params_df, model_spec_init_dict)


if __name__ == "__main__":

    minutes = float(sys.argv[1])
    func(datetime.timedelta(minutes=0.1))

cleanup()
