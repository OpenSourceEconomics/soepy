import jax.numpy as jnp
import numpy as np
import pytest

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.solve.continuous_continuation import (
    interpolate_then_weight_continuation_values,
)
from soepy.solve.terminal_proxy import terminal_proxy_continuation


def manual_linear_interp(grid, values, x):
    x_lo, x_hi = grid[0], grid[1]
    v_lo, v_hi = values[0], values[1]
    weight = (x - x_lo) / (x_hi - x_lo)
    return v_lo + weight * (v_hi - v_lo)


@pytest.fixture(name="continuation_out")
def fixture_continuation_out():
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

    prob_child = 1 / 4
    prob_partner_states = jnp.array([2 / 5, 3 / 5])

    init_exp_max = 1.0

    model_params = type(
        "Params",
        (),
        {
            "gamma_p": jnp.array([0.5]),
            "gamma_p_mom": 0.0,
            "exp_depr_rate": jnp.array([0.1]),
        },
    )()

    out = interpolate_then_weight_continuation_values(
        exp_grid=exp_grid,
        v_next_grid=v_next_grid,
        child_state_indexes_local=child_state_indexes_local,
        period=0,
        init_exp_max=init_exp_max,
        model_params=model_params,
        educ_level=jnp.array([0]),
        child_age=jnp.array([-1]),
        biased_exp=False,
        prob_child_states=jnp.array([prob_child]),
        prob_partner_states=jnp.array([prob_partner_states]),
    )

    return {
        "exp_grid": exp_grid,
        "v_next_grid": v_next_grid,
        "prob_child": prob_child,
        "prob_partner_states": prob_partner_states,
        "out": out,
        "init_exp_max": init_exp_max,
    }


def test_interpolate_then_weight_continuation_values_choice0_shape_and_value(
    continuation_out,
):
    exp_grid = continuation_out["exp_grid"]
    out = continuation_out["out"]
    v_next_grid = continuation_out["v_next_grid"]

    assert out.shape == (1, 3, 2)

    depr_rate = 0.1
    exp_years_ne = exp_grid * 2
    exp_years_ne_next = exp_years_ne * (1 - depr_rate)
    x_next_ne = exp_years_ne_next / 3
    expected = manual_linear_interp(exp_grid, v_next_grid[0], x_next_ne)
    np.testing.assert_allclose(out[0, 0], expected)


def test_interpolate_then_weight_continuation_values_choice1_shape_and_value(
    continuation_out,
):
    exp_grid = continuation_out["exp_grid"]
    out = continuation_out["out"]
    v_next_grid = continuation_out["v_next_grid"]

    assert out.shape == (1, 3, 2)

    depr_rate = 0.1
    exp_years_pt = exp_grid * 2
    exp_years_pt_next = exp_years_pt * (1 - depr_rate) + 0.5
    x_next_pt = exp_years_pt_next / 3
    expected = manual_linear_interp(exp_grid, v_next_grid[0], x_next_pt)
    np.testing.assert_allclose(out[0, 1], expected)


def test_interpolate_then_weight_continuation_values_choice2_shape_and_value(
    continuation_out,
):
    exp_grid = continuation_out["exp_grid"]
    out = continuation_out["out"]
    v_next_grid = continuation_out["v_next_grid"]
    prob_child = continuation_out["prob_child"]
    prob_partner_states = continuation_out["prob_partner_states"]

    assert out.shape == (1, 3, 2)

    # Scale factor for interpolation with init_exp_max
    depr_rate = 0.1
    exp_years_ft = exp_grid * 2
    exp_years_ft_next = exp_years_ft * (1 - depr_rate) + 1
    x_next_ft = exp_years_ft_next / (2 * continuation_out["init_exp_max"] + 1)
    prob_single, prob_partner = prob_partner_states
    val_no_child_single = manual_linear_interp(exp_grid, v_next_grid[0], x_next_ft)
    val_no_child_partner = manual_linear_interp(exp_grid, v_next_grid[1], x_next_ft)
    val_child_single = manual_linear_interp(exp_grid, v_next_grid[2], x_next_ft)
    val_child_partner = manual_linear_interp(exp_grid, v_next_grid[0], x_next_ft)
    expected = (1 - prob_child) * (
        prob_single * val_no_child_single + prob_partner * val_no_child_partner
    ) + prob_child * (prob_single * val_child_single + prob_partner * val_child_partner)
    np.testing.assert_allclose(out[0, 2], expected)


def test_terminal_proxy_shape_and_monotonicity():
    exp_grid = jnp.array([0.0, 1.0])
    states_period = jnp.zeros((4, 6), dtype=int)
    states_period = states_period.at[:, EDUC_LEVEL].set(jnp.array([0, 2, 0, 0]))
    states_period = states_period.at[:, AGE_YOUNGEST_CHILD].set(-1)

    covariates_period = jnp.zeros((4, 4), dtype=float)
    covariates_period = covariates_period.at[0, 1].set(1.0)
    covariates_period = covariates_period.at[1, 1].set(1.0)
    covariates_period = covariates_period.at[2, 1].set(1.0)
    covariates_period = covariates_period.at[3, 1].set(2.0)

    model_params = type(
        "Params",
        (),
        {
            "gamma_0": jnp.array([0.0, 0.5, 1.0]),
            "gamma_1": jnp.array([0.2, 0.2, 0.2]),
            "gamma_p": jnp.array([1.0, 1.0, 1.0]),
            "gamma_p_mom": 0.0,
            "exp_depr_rate": jnp.array([0.0, 0.0, 0.0]),
            "beta_0": 0.0,
            "beta_1": 1.0,
            "beta_2": 1.0,
        },
    )()

    model_spec = type(
        "Spec",
        (),
        {
            "init_exp_max": 1.0,
            "elasticity_scale": 1.0,
        },
    )()

    out = terminal_proxy_continuation(
        exp_grid=exp_grid,
        states_period=states_period,
        covariates_period=covariates_period,
        model_params=model_params,
        model_spec=model_spec,
        biased_exp=False,
        current_period=0,
    )

    assert out.shape == (4, NUM_CHOICES, 2)
    # Negative monotinicity. In the estimation betas will be negative
    assert np.all(out < 0)
    assert np.all(out[1] < out[0])
    assert np.allclose(out[3], out[2])
    assert np.all(out[0, 2, 1] < out[0, 2, 0])
