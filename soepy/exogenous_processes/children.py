"""This module reads in information on probabilities regarding the exogenous
process of childbirth."""
import numpy as np
import pandas as pd


def define_child_age_update_rule(model_spec, states, covariates):
    """Defines a vector with the length of the number of states that contains the
    value the state space component `age_kid` should take depending on whether or not
     a child arrives in the period.
     The purpose of this object is to facilitate easy child-parent state look-up
     in the backward induction."""

    # Child arrives is equivalent to new child age = 0
    # If no child arrives we need to specify an update rule

    # Age stays at -1 if no kids so far
    child_age_update_rule = np.full(states.shape[0], -1)
    # Age increases by one, if there is a kid
    child_age_update_rule[np.where(covariates[:, 0] != 0)] = (
        states[np.where(covariates[:, 0] != 0)][:, 6] + 1
    )
    # Age does not exceed 10.
    child_age_update_rule[
        child_age_update_rule > model_spec.child_age_max
    ] = model_spec.child_age_max

    return child_age_update_rule


def gen_prob_child_vector(model_spec):
    """Generates a vector with length `num_periods` which contains
    the probability to get a child in the corresponding period."""

    # Read data frame with information on probability to get a child
    # in every period
    exog_child_info_df = pd.read_pickle(model_spec.child_info_file_name)

    exog_child_info_df = exog_child_info_df.iloc[
        exog_child_info_df.index.get_level_values("period") < model_spec.num_periods
    ]

    prob_child_values = exog_child_info_df["prob_child_values"].to_numpy()

    prob_child = np.full(model_spec.num_periods, 0.00)
    prob_child[
        0 : min(model_spec.last_child_bearing_period + 1, model_spec.num_periods)
    ] += prob_child_values[
        0 : min(model_spec.last_child_bearing_period + 1, model_spec.num_periods)
    ]

    # Assert length of array equals num periods
    assert (
        len(prob_child) == model_spec.num_periods
    ), "Probability of childbirth and number of periods length mismatch"

    return prob_child


def gen_prob_child_init_age_vector(model_spec):
    """Generates a list of lists containing the shares of individuals with
    kids aged -1 (no kids), 0, 1, 2, 3, and 4 in the model's first period.
    Shares differ by the level of education of the individuals."""

    child_age_shares = pd.read_pickle(model_spec.child_age_shares_file_name)

    prob_child_age = []
    for educ_level in range(model_spec.num_educ_levels):
        child_age_shares_list = child_age_shares[
            child_age_shares.index.get_level_values("educ_level") == educ_level
        ]["child_age_shares"].to_list()
        child_age_shares_list[0] = 1 - sum(child_age_shares_list[1:])
        prob_child_age.append(child_age_shares_list)

    return prob_child_age
