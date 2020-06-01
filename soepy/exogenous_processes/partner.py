"""This module reads in information on probabilities regarding the exogenous
process of marriage."""
import pandas as pd


def gen_prob_partner(model_spec):
    """ Generates a vector with length `num_periods` which contains
    the probability to get a partner in the corresponding period."""

    # Read data frame with information on probability to get a partner
    # in every period

    exog_partner_info_df = pd.read_pickle(model_spec.partner_info_file_name)

    prob_partner = exog_partner_info_df.values.reshape(model_spec.num_periods, 3)

    # Assert length of array equals num periods
    assert (
        len(prob_partner) == model_spec.num_periods
    ), "Probability of marriage and number of periods length mismatch"

    return prob_partner
