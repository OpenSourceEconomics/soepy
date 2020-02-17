import numpy as np

from soepy.shared.shared_constants import LAST_CHILD_BEARING_PERIOD, KIDS_AGES
from soepy.exogenous_processes.sample_data import PROB_CHILD_VALUES


def define_child_age_update_rule(states, covariates):
    """ Defines a vector with the length of the number of states that contains the
    value the state space component `age_kid` should take depending on whether or not
     a child arrives in the period.
     The purpose of this object is to facilitate easy child-parent state look-up
     in the backward induction."""

    # Child arrives is equivalent to new child age = 0
    # If no child arrives we need to specify an update rule

    # Age stays at -1 if no kids so far
    child_age_update_rule = np.full(states.shape[0], -1)
    # Age increases by one, if there is a kid
    child_age_update_rule[np.where(covariates[:, 1] != 0)] = (
        states[np.where(covariates[:, 1] != 0)][:, 6] + 1
    )
    # Age does not exceed 11.
    child_age_update_rule[child_age_update_rule > max(KIDS_AGES)] = 11

    return child_age_update_rule


def gen_prob_child_vector(
    model_spec,
    LAST_CHILD_BEARING_PERIOD=LAST_CHILD_BEARING_PERIOD,
    PROB_CHILD_VALUES=PROB_CHILD_VALUES,
):
    """ Generates a vector with length `num_periods` which contains
    the probability to get a child in the corresponding period."""

    prob_child = np.full(model_spec.num_periods, 0.00)
    prob_child[
        0 : min(LAST_CHILD_BEARING_PERIOD + 1, model_spec.num_periods)
    ] += PROB_CHILD_VALUES[
        0 : min(LAST_CHILD_BEARING_PERIOD + 1, model_spec.num_periods)
    ]

    return prob_child
