"""This module reads in information on probabilities regarding the exogenous
process of childbirth."""
import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD


def define_child_age_update_rule(model_spec, states):
    """Define next-period child age under the no-new-child branch."""

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
    exog_child_info_df = exog_child_info_df.sort_index()

    idx_names = set(exog_child_info_df.index.names)
    has_partner = "partner_present" in idx_names
    has_prior_kid = "has_prior_kid" in idx_names
    has_child_state = "child_state" in idx_names
    has_kids = "has_kids" in idx_names
    educ_count = exog_child_info_df.index.get_level_values("educ_level").nunique()

    if has_partner and has_child_state:
        child_state_count = exog_child_info_df.index.get_level_values(
            "child_state"
        ).nunique()
        prob_child_values = exog_child_info_df.values.reshape(
            model_spec.num_periods, educ_count, 2, child_state_count
        )
        assert prob_child_values.shape[0] == model_spec.num_periods
        return prob_child_values

    if has_partner and has_kids:
        prob_child_values = exog_child_info_df.values.reshape(
            model_spec.num_periods, educ_count, 2, 2
        )
        assert prob_child_values.shape[0] == model_spec.num_periods
        return prob_child_values

    if has_partner and has_prior_kid:
        prob_child_values = exog_child_info_df.values.reshape(
            model_spec.num_periods, educ_count, 2, 2
        )
        assert prob_child_values.shape[0] == model_spec.num_periods
        return prob_child_values

    if has_child_state:
        child_state_count = exog_child_info_df.index.get_level_values(
            "child_state"
        ).nunique()
        prob_child_values = exog_child_info_df.values.reshape(
            model_spec.num_periods, educ_count, child_state_count
        )
        assert prob_child_values.shape[0] == model_spec.num_periods
        return prob_child_values

    prob_child_values = exog_child_info_df.values.reshape(
        model_spec.num_periods, educ_count
    )
    assert prob_child_values.shape[0] == model_spec.num_periods
    return prob_child_values
