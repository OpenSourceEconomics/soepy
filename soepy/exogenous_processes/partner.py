"""This module reads in information on probabilities regarding the exogenous
process of marriage."""
import pandas as pd
import numpy as np


def gen_prob_partner(model_spec):
    """ Generates a vector with length `num_periods` which contains
    the probability to get a partner in the corresponding period."""

    # Read data frame with information on probability to get a partner
    # in every period
    if model_spec.partner_info_file_name == "test":
        index_levels = [list(range(0, model_spec.num_periods)), [0, 1, 2]]

        index = pd.MultiIndex.from_product(index_levels, names=["period", "educ_level"])
        exog_partner_info_df = pd.DataFrame(
            np.random.uniform(0, 1, size=model_spec.num_periods * 3).tolist(),
            index=index,
            columns=["exog_partner_values"],
        )
    else:
        exog_partner_info_df = pd.read_pickle(model_spec.partner_info_file_name)

    prob_partner = exog_partner_info_df.values.reshape(model_spec.num_periods, 3)

    # Assert length of array equals num periods
    assert (
        len(prob_partner) == model_spec.num_periods
    ), "Probability of marriage and number of periods length mismatch"

    return prob_partner
