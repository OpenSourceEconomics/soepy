import collections

import numpy as np
import pytest

from soepy.solve.create_state_space import pyth_create_state_space


@pytest.fixture(scope="module")
def created_state_space():
    """This test ensures that the state space creation generates the correct admissible
    state space points for the first 4 periods."""
    model_spec = collections.namedtuple(
        "model_spec",
        "num_periods num_educ_levels num_types \
         last_child_bearing_period, child_age_max \
         educ_years child_age_init_max init_exp_max",
    )
    model_spec = model_spec(2, 3, 2, 24, 10, [0, 0, 0], 4, 2)

    states, _ = pyth_create_state_space(model_spec)

    return states


def test_state_space_shape(created_state_space):
    states_shape_true = (2916, 8)
    np.testing.assert_array_equal(states_shape_true, created_state_space.shape)


def test_state_space_batch_1(created_state_space):
    states_batch_1_true = [
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 2, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 1, 0, -1, 0],
        [0, 0, 0, 2, 1, 0, -1, 0],
        [0, 0, 0, 0, 2, 0, -1, 0],
        [0, 0, 0, 1, 2, 0, -1, 0],
        [0, 0, 0, 2, 2, 0, -1, 0],
        [0, 1, 0, 0, 0, 0, -1, 0],
        [0, 1, 0, 1, 0, 0, -1, 0],
        [0, 1, 0, 2, 0, 0, -1, 0],
        [0, 1, 0, 0, 1, 0, -1, 0],
        [0, 1, 0, 1, 1, 0, -1, 0],
        [0, 1, 0, 2, 1, 0, -1, 0],
        [0, 1, 0, 0, 2, 0, -1, 0],
        [0, 1, 0, 1, 2, 0, -1, 0],
        [0, 1, 0, 2, 2, 0, -1, 0],
        [0, 2, 0, 0, 0, 0, -1, 0],
        [0, 2, 0, 1, 0, 0, -1, 0],
        [0, 2, 0, 2, 0, 0, -1, 0],
        [0, 2, 0, 0, 1, 0, -1, 0],
        [0, 2, 0, 1, 1, 0, -1, 0],
        [0, 2, 0, 2, 1, 0, -1, 0],
        [0, 2, 0, 0, 2, 0, -1, 0],
        [0, 2, 0, 1, 2, 0, -1, 0],
        [0, 2, 0, 2, 2, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
    ]
    np.testing.assert_array_equal(states_batch_1_true, created_state_space[0:30])


def test_state_space_batch_2(created_state_space):
    states_batch_2_true = [
        [1, 0, 1, 3, 0, 0, -1, 1],
        [1, 0, 0, 0, 1, 0, -1, 1],
        [1, 0, 2, 0, 1, 0, -1, 1],
        [1, 0, 0, 1, 1, 0, -1, 1],
        [1, 0, 1, 1, 1, 0, -1, 1],
        [1, 0, 2, 1, 1, 0, -1, 1],
        [1, 0, 0, 2, 1, 0, -1, 1],
        [1, 0, 1, 2, 1, 0, -1, 1],
        [1, 0, 2, 2, 1, 0, -1, 1],
        [1, 0, 1, 3, 1, 0, -1, 1],
        [1, 0, 0, 0, 2, 0, -1, 1],
        [1, 0, 2, 0, 2, 0, -1, 1],
        [1, 0, 0, 1, 2, 0, -1, 1],
        [1, 0, 1, 1, 2, 0, -1, 1],
        [1, 0, 2, 1, 2, 0, -1, 1],
        [1, 0, 0, 2, 2, 0, -1, 1],
        [1, 0, 1, 2, 2, 0, -1, 1],
        [1, 0, 2, 2, 2, 0, -1, 1],
        [1, 0, 1, 3, 2, 0, -1, 1],
        [1, 0, 2, 0, 3, 0, -1, 1],
        [1, 0, 2, 1, 3, 0, -1, 1],
        [1, 0, 2, 2, 3, 0, -1, 1],
        [1, 1, 0, 0, 0, 0, -1, 1],
        [1, 1, 0, 1, 0, 0, -1, 1],
        [1, 1, 1, 1, 0, 0, -1, 1],
        [1, 1, 0, 2, 0, 0, -1, 1],
        [1, 1, 1, 2, 0, 0, -1, 1],
        [1, 1, 1, 3, 0, 0, -1, 1],
        [1, 1, 0, 0, 1, 0, -1, 1],
        [1, 1, 2, 0, 1, 0, -1, 1],
    ]
    np.testing.assert_array_equal(states_batch_2_true, created_state_space[1220:1250])


def test_state_space_batch_3(created_state_space):

    states_batch_3_true = [
        [1, 2, 2, 0, 2, 1, 0, 1],
        [1, 2, 0, 1, 2, 1, 0, 1],
        [1, 2, 1, 1, 2, 1, 0, 1],
        [1, 2, 2, 1, 2, 1, 0, 1],
        [1, 2, 0, 2, 2, 1, 0, 1],
        [1, 2, 1, 2, 2, 1, 0, 1],
        [1, 2, 2, 2, 2, 1, 0, 1],
        [1, 2, 1, 3, 2, 1, 0, 1],
        [1, 2, 2, 0, 3, 1, 0, 1],
        [1, 2, 2, 1, 3, 1, 0, 1],
        [1, 2, 2, 2, 3, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 2, 0, 1, 1, 1],
        [1, 0, 1, 2, 0, 1, 1, 1],
        [1, 0, 1, 3, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 2, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 2, 1, 1, 1, 1, 1],
        [1, 0, 0, 2, 1, 1, 1, 1],
        [1, 0, 1, 2, 1, 1, 1, 1],
        [1, 0, 2, 2, 1, 1, 1, 1],
        [1, 0, 1, 3, 1, 1, 1, 1],
        [1, 0, 0, 0, 2, 1, 1, 1],
        [1, 0, 2, 0, 2, 1, 1, 1],
        [1, 0, 0, 1, 2, 1, 1, 1],
        [1, 0, 1, 1, 2, 1, 1, 1],
    ]

    np.testing.assert_array_equal(states_batch_3_true, created_state_space[2500:2530])
