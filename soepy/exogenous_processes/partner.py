import numpy as np
import pandas as pd


def gen_prob_partner(model_spec):
    exog_partner_separation_info_df = pd.read_pickle(
        model_spec.partner_separation_info_file_name
    )

    exog_partner_separation_info_df = exog_partner_separation_info_df.loc[
        (range(model_spec.num_periods), slice(None)), :
    ]

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
