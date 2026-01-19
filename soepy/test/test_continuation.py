import jax.numpy as jnp
import numpy as np

from soepy.solve.continuous_continuation import (
    interpolate_then_weight_continuation_values,
)


def test_interpolate_then_weight_continuation_values_choice2_shape_and_value():
    exp_grid = jnp.array([0.0, 1.0])

    v_next_grid = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )

    child_state_indexes_local = jnp.array(
        [
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 1], [2, 0]],
            ]
        ]
    )

    out = interpolate_then_weight_continuation_values(
        exp_grid=exp_grid,
        v_next_grid=v_next_grid,
        child_state_indexes_local=child_state_indexes_local,
        period=0,
        init_exp_max=1,
        pt_increment_states=jnp.array([0.5]),
        prob_child_states=jnp.array([0.25]),
        prob_partner_states=jnp.array([[0.4, 0.6]]),
    )

    assert out.shape == (1, 3, 2)

    expected = np.array([0.825, 1.65])
    np.testing.assert_allclose(out[0, 2], expected)
