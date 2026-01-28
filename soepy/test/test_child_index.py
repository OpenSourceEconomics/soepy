import collections

import numpy as np
import pytest

from soepy.shared.constants_and_indices import MISSING_INT
from soepy.solve.create_state_space import create_state_space_objects


def _age_to_idx(age, n_kid_ages):
    return n_kid_ages - 1 if age == -1 else age


@pytest.fixture()
def model_spec():
    spec = {
        "num_periods": 4,
        "num_educ_levels": 2,
        "num_types": 2,
        "child_age_max": 2,
        "partner_cf_const": 0.0,
        "partner_cf_age": 0.0,
        "partner_cf_age_sq": 0.0,
        "partner_cf_educ": 0.0,
        "child_benefits": 0.0,
    }
    return collections.namedtuple("model_specification", spec.keys())(**spec)


def test_child_state_indexes_match_indexer(model_spec):
    (
        states,
        indexer,
        _,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec=model_spec)

    n_kid_ages = indexer.shape[4]

    # Check mapping for all non-terminal states.
    for state_idx, state in enumerate(states):
        period, educ_level, _, type_, age, partner = state

        if period == model_spec.num_periods - 1:
            assert (child_state_indexes[state_idx] == MISSING_INT).all()
            continue

        next_period = period + 1
        next_age_no_child = child_age_update_rule[state_idx]

        for choice in range(3):
            for child_arrival in [0, 1]:
                for next_partner in [0, 1]:
                    expected_age = 0 if child_arrival == 1 else next_age_no_child
                    expected = indexer[
                        next_period,
                        educ_level,
                        choice,
                        type_,
                        _age_to_idx(int(expected_age), n_kid_ages),
                        next_partner,
                    ]
                    got = child_state_indexes[
                        state_idx, choice, child_arrival, next_partner
                    ]
                    assert got == expected


def test_child_state_indexes_terminal_all_missing(model_spec):
    states, _, _, _, child_state_indexes = create_state_space_objects(
        model_spec=model_spec
    )

    terminal = states[:, 0] == model_spec.num_periods - 1
    assert terminal.any()
    assert (child_state_indexes[terminal] == MISSING_INT).all()
