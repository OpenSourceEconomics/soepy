"""This module reads in information on probabilities regarding the exogenous
process of childbirth."""
import numpy as np
import pandas as pd

from soepy.shared.state_space_indices import AGE_YOUNGEST_CHILD


def define_child_age_update_rule(model_spec, states):
    """Define next-period child age under the no-new-child branch."""

    child_age_update_rule = np.full(states.shape[0], -1, dtype=np.int32)

    has_kid = states[:, AGE_YOUNGEST_CHILD] != -1
    child_age_update_rule[has_kid] = states[has_kid][:, AGE_YOUNGEST_CHILD] + 1

    child_age_update_rule[child_age_update_rule > model_spec.child_age_max] = -1
    return child_age_update_rule


def gen_prob_child_vector(model_spec):
    """Generate probability of childbirth for each period and lagged choice."""

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
    """Generate shares of initial child ages by education level."""

    child_age_shares = pd.read_pickle(model_spec.child_age_shares_file_name)

    prob_child_age = []
    for educ_level in range(model_spec.num_educ_levels):
        child_age_shares_list = child_age_shares[
            child_age_shares.index.get_level_values("educ_level") == educ_level
        ]["child_age_shares"].to_list()
        child_age_shares_list[0] = 1 - sum(child_age_shares_list[1:])
        prob_child_age.append(child_age_shares_list)

    return prob_child_age
