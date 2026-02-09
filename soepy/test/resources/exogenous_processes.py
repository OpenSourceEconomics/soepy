"""Test-only utilities for exogenous process probabilities."""
import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD


def define_child_age_update_rule(model_spec, states):
    child_age_update_rule = np.full(states.shape[0], -1, dtype=np.int32)

    has_kid = states[:, AGE_YOUNGEST_CHILD] != -1
    child_age_update_rule[has_kid] = states[has_kid][:, AGE_YOUNGEST_CHILD] + 1

    child_age_update_rule[child_age_update_rule > model_spec.child_age_max] = -1
    return child_age_update_rule


def gen_prob_child_vector(model_spec):
    exog_child_info_df = pd.read_pickle(model_spec.child_info_file_name)

    exog_child_info_df = exog_child_info_df.iloc[
        exog_child_info_df.index.get_level_values("period") < model_spec.num_periods
    ]

    prob_child_values = exog_child_info_df.values.reshape(model_spec.num_periods, 3)

    prob_child = np.full((model_spec.num_periods, 3), 0.00)
    prob_child[
        0 : min(model_spec.last_child_bearing_period + 1, model_spec.num_periods)
    ] += prob_child_values[
        0 : min(model_spec.last_child_bearing_period + 1, model_spec.num_periods)
    ]

    assert len(prob_child) == model_spec.num_periods
    return prob_child


def gen_prob_child_init_age_vector(model_spec):
    child_age_shares = pd.read_pickle(model_spec.child_age_shares_file_name)

    prob_child_age = []
    for educ_level in range(model_spec.num_educ_levels):
        child_age_shares_list = child_age_shares[
            child_age_shares.index.get_level_values("educ_level") == educ_level
        ]["child_age_shares"].to_list()
        child_age_shares_list[0] = 1 - sum(child_age_shares_list[1:])
        prob_child_age.append(child_age_shares_list)

    return prob_child_age


def gen_prob_educ_level_vector(model_spec):
    prob_educ_level = pd.read_pickle(model_spec.educ_shares_file_name)

    prob_educ_level = list(prob_educ_level["educ_shares"])

    return prob_educ_level


def gen_prob_init_exp_component_vector(model_spec, model_spec_exp_file_key):
    exp_shares = pd.read_pickle(model_spec_exp_file_key)

    init_exp = []
    for educ_level in range(model_spec.num_educ_levels):
        exp_shares_list = exp_shares[
            exp_shares.index.get_level_values("educ_level") == educ_level
        ]["exper_shares"].to_list()
        exp_shares_list[0] = 1 - sum(exp_shares_list[1:])
        init_exp.append(exp_shares_list)

    return init_exp


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


def gen_prob_partner_present_vector(model_spec):
    partner_shares = pd.read_pickle(model_spec.partner_shares_file_name)
    return partner_shares.to_numpy().flatten()
