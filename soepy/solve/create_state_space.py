import numpy as np

from soepy.exogenous_processes.children import define_child_age_update_rule
from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import LAGGED_CHOICE
from soepy.shared.constants_and_indices import MISSING_INT
from soepy.shared.constants_and_indices import N_STATE_VARS
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.constants_and_indices import PARTNER
from soepy.shared.constants_and_indices import PERIOD
from soepy.shared.constants_and_indices import TYPE
from soepy.solve.covariates import construct_covariates


def create_state_space_objects(model_spec):
    """Create state space and associated lookup/transition objects.

    This repository now uses a continuous experience stock. Therefore the discrete state
    space does not contain part-time or full-time experience dimensions.

    Returns
    -------
    states : np.ndarray
        Discrete state array of shape (n_states, 6).
    indexer : np.ndarray
        Indexer array of shape (num_periods, num_educ_levels, NUM_CHOICES, num_types,
        n_kids_ages, 2).
    covariates : np.ndarray
        Covariates derived from the discrete state.
    child_age_update_rule : np.ndarray
        Next-period child age under "no new child".
    child_state_indexes : np.ndarray
        Indices of next-period discrete states for each choice and child/partner branch.
        Shape (n_states, NUM_CHOICES, 2, 2).
    """

    states, indexer = create_discrete_state_space(model_spec)
    covariates = construct_covariates(states, model_spec)

    child_age_update_rule = define_child_age_update_rule(model_spec, states)
    child_state_indexes = create_child_indexes(
        states=states,
        indexer=indexer,
        model_spec=model_spec,
        child_age_update_rule=child_age_update_rule,
    )

    return states, indexer, covariates, child_age_update_rule, child_state_indexes


def create_discrete_state_space(model_spec):
    """Create the discrete part of the state space.

    State layout (columns):
    - period
    - educ_level
    - lagged_choice
    - type
    - age_youngest_child
    - partner

    The experience stock is handled separately via a continuous grid.
    """

    kids_ages = np.arange(-1, model_spec.child_age_max + 1, dtype=np.int32)
    n_kids_ages = kids_ages.shape[0]

    indexer_shape = (
        model_spec.num_periods,
        model_spec.num_educ_levels,
        NUM_CHOICES,
        model_spec.num_types,
        n_kids_ages,
        2,
    )
    indexer = np.full(indexer_shape, MISSING_INT, dtype=np.int32)

    blocks = []
    i = 0

    for period in range(model_spec.num_periods):
        for educ_level in range(model_spec.num_educ_levels):
            for type_ in range(model_spec.num_types):
                for partner in range(2):
                    for age_kid in kids_ages:
                        age_idx = _kid_age_to_index(int(age_kid), n_kids_ages)

                        lagged = np.arange(NUM_CHOICES, dtype=np.int32)
                        n = lagged.size

                        block = np.empty((n, N_STATE_VARS), dtype=np.int32)
                        block[:, PERIOD] = period
                        block[:, EDUC_LEVEL] = educ_level
                        block[:, LAGGED_CHOICE] = lagged
                        block[:, TYPE] = type_
                        block[:, AGE_YOUNGEST_CHILD] = age_kid
                        block[:, PARTNER] = partner

                        blocks.append(block)

                        ids = np.arange(i, i + n, dtype=np.int32)
                        indexer[
                            period,
                            educ_level,
                            lagged,
                            type_,
                            age_idx,
                            partner,
                        ] = ids
                        i += n

    states = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, N_STATE_VARS), dtype=np.int32)
    )

    return states, indexer


def create_child_indexes(states, indexer, model_spec, child_age_update_rule):
    """Create child-state indices for the discrete transition.

    Output shape: (n_states, NUM_CHOICES, 2, 2) filled with MISSING_INT where invalid.

    Branch meaning (axis 2 and 3):
    - axis 2: child arrives (1) vs no child arrives (0)
    - axis 3: partner next period is 0 vs 1

    Experience transitions are handled separately via interpolation.
    """

    child_indexes = np.full(
        (states.shape[0], NUM_CHOICES, 2, 2), MISSING_INT, dtype=np.int32
    )
    if states.shape[0] == 0:
        return child_indexes

    period = states[:, PERIOD]
    educ = states[:, EDUC_LEVEL]
    lagged_choice = states[:, LAGGED_CHOICE]
    type_ = states[:, TYPE]
    age_kid_val = states[:, AGE_YOUNGEST_CHILD]
    partner = states[:, PARTNER]

    n_kid_ages = indexer.shape[4]
    age_idx = np.where(age_kid_val == -1, n_kid_ages - 1, age_kid_val)

    parent_idx = np.where(period < (model_spec.num_periods - 1))[0]
    next_period = period[parent_idx] + 1

    k_parent = indexer[
        period[parent_idx],
        educ[parent_idx],
        lagged_choice[parent_idx],
        type_[parent_idx],
        age_idx[parent_idx],
        partner[parent_idx],
    ]
    update_rule = child_age_update_rule[k_parent]

    for choice in range(NUM_CHOICES):
        child_indexes[parent_idx, choice, 0, 0] = indexer[
            next_period,
            educ[parent_idx],
            choice,
            type_[parent_idx],
            update_rule,
            0,
        ]
        child_indexes[parent_idx, choice, 0, 1] = indexer[
            next_period,
            educ[parent_idx],
            choice,
            type_[parent_idx],
            update_rule,
            1,
        ]

        child_indexes[parent_idx, choice, 1, 0] = indexer[
            next_period,
            educ[parent_idx],
            choice,
            type_[parent_idx],
            0,
            0,
        ]
        child_indexes[parent_idx, choice, 1, 1] = indexer[
            next_period,
            educ[parent_idx],
            choice,
            type_[parent_idx],
            0,
            1,
        ]

    return child_indexes


def _kid_age_to_index(age_kid: int, kids_ages_len: int) -> int:
    """Map child age value to indexer position.

    - age_kid == -1 -> last element
    - else -> age_kid
    """

    return kids_ages_len - 1 if age_kid == -1 else age_kid
