"""This module reads in information on probabilities regarding the exogenous
process of marriage."""
import numpy as np
import pandas as pd


def gen_prob_partner(model_spec):
    """Generates a vector with length `num_periods` which contains
    the probability to loose ones partner in the corresponding period."""

    # Read data frame with information on probability to loose ones partner
    # in every period
    exog_partner_separation_info_df = pd.read_pickle(
        model_spec.partner_separation_info_file_name
    )

    exog_partner_separation_info_df = exog_partner_separation_info_df.loc[
        (range(model_spec.num_periods), slice(None)), :
    ]

    # Read data frame with information on probability to get a partner
    # in every period
    exog_partner_arrival_info_df = pd.read_pickle(
        model_spec.partner_arrival_info_file_name
    )

    exog_partner_arrival_info_df = exog_partner_arrival_info_df.loc[
        (range(model_spec.num_periods), slice(None)), :
    ]

    prob_mat = np.zeros((model_spec.num_periods, model_spec.num_educ_levels, 2, 2))
    prob_mat[:, :, 0, 1] = exog_partner_arrival_info_df.values.reshape(
        model_spec.num_periods, model_spec.num_educ_levels
    )
    prob_mat[:, :, 0, 0] = 1 - prob_mat[:, :, 0, 1]
    prob_mat[:, :, 1, 0] = exog_partner_separation_info_df.values.reshape(
        model_spec.num_periods, model_spec.num_educ_levels
    )
    prob_mat[:, :, 1, 1] = 1 - prob_mat[:, :, 1, 0]

    return prob_mat


def gen_prob_partner_present_vector(model_spec):
    """Generates a list containing the shares of individuals with
    a partner present in the household in the model's first period.
    Shares differ by the level of education of the individuals."""

    partner_shares = pd.read_pickle(model_spec.partner_shares_file_name)
    return partner_shares.to_numpy().flatten()
