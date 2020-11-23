"""This module reads in information on probabilities regarding the exogenous
process of marriage."""
import pandas as pd


def gen_prob_partner_arrival(model_spec):
    """Generates a vector with length `num_periods` which contains
    the probability to get a partner in the corresponding period."""

    # Read data frame with information on probability to get a partner
    # in every period
    exog_partner_arrival_info_df = pd.read_pickle(
        model_spec.partner_arrival_info_file_name
    )

    exog_partner_arrival_info_df = exog_partner_arrival_info_df.iloc[
        exog_partner_arrival_info_df.index.get_level_values("period")
        < model_spec.num_periods
    ]

    prob_partner_arrival = exog_partner_arrival_info_df.values.reshape(
        model_spec.num_periods, 3
    )

    # Assert length of array equals num periods
    assert (
        len(prob_partner_arrival) == model_spec.num_periods
    ), "Probability of marriage and number of periods length mismatch"

    return prob_partner_arrival
