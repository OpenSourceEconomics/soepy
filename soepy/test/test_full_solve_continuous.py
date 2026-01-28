import collections

import jax.numpy as jnp
import numpy as np

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import PARTNER
from soepy.shared.constants_and_indices import PERIOD
from soepy.shared.experience_stock import get_pt_increment
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve


def _make_min_model_spec():
    # Keep this minimal but complete for solve + covariates + emax.
    spec = {
        # State space
        "num_periods": 5,
        "num_educ_levels": 1,
        "num_types": 1,
        "child_age_max": 0,
        "init_exp_max": 1,
        # Continuous experience
        "experience_grid_points": 10,
        "pt_exp_ratio": 0.5,
        # Covariates
        "partner_cf_const": -1e9,
        "partner_cf_age": 0.0,
        "partner_cf_age_sq": 0.0,
        "partner_cf_educ": 0.0,
        "child_benefits": 0.0,
        # EMAX integration
        "integration_method": "quadrature",
        "num_draws_emax": 1,
        "seed_emax": 0,
        # Taxes/transfers (set so taxes are always zero)
        "ssc_deductions": jnp.array([0.0, 1e9]),
        "tax_params": jnp.array(
            [
                [1e9, 1e9, 1e9, 1e9],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "tax_splitting": True,
        "child_care_costs": jnp.zeros((3, 2)),
        # Preferences / benefits
        "parental_leave_regime": "elterngeld",
        "alg1_replacement_no_child": 0.0,
        "alg1_replacement_child": 0.0,
        "regelsatz_single": 0.0,
        "housing_single": 0.0,
        "housing_addtion": 0.0,
        "regelsatz_child": 0.0,
        "addition_child_single": 0.0,
        "elterngeld_replacement": 0.0,
        "elterngeld_min": 0.0,
        "elterngeld_max": 0.0,
        "erziehungsgeld_income_threshold_single": 1e9,
        "erziehungsgeld_income_threshold_married": 1e9,
        "erziehungsgeld": 0.0,
        # Wage scaling
        "elasticity_scale": 1.0,
        "exp_grid": jnp.linspace(0.0, 1.0, 10),
    }

    return collections.namedtuple("model_specification", spec.keys())(**spec)


def _make_min_model_params():
    # One education group, one type.
    params = {
        "shock_sd": 0.0,
        "delta": 0.95,
        "mu": 1.0,
        # Wages: log_wage = gamma_0 + gamma_f * log(exp_years + 1)
        "gamma_0": jnp.array([0.0]),
        "gamma_1": jnp.array([0.5]),
        "gamma_p": jnp.array([0.3]),
        "gamma_p_mom": 0.0,
        # Non-consumption utility parameters (set to 0 => exp(0)=1)
        "theta_p": jnp.array([0.0]),
        "theta_f": jnp.array([0.0]),
        "no_kids_f": jnp.array([0.0]),
        "no_kids_p": jnp.array([0.0]),
        "yes_kids_f": jnp.array([0.0]),
        "yes_kids_p": jnp.array([0.0]),
        "child_0_2_f": 0.0,
        "child_0_2_p": 0.0,
        "child_3_5_f": 0.0,
        "child_3_5_p": 0.0,
        "child_6_10_f": 0.0,
        "child_6_10_p": 0.0,
    }

    return collections.namedtuple("model_parameters", params.keys())(**params)


def _next_stock_np(x, period, init_exp_max, pt_inc, choice):
    max_years_t = init_exp_max + max(period, period * pt_inc)
    exp_years = x * max_years_t

    exp_years_next = exp_years + (choice == 2) * 1.0 + (choice == 1) * pt_inc

    max_years_tp1 = init_exp_max + max(period + 1, (period + 1) * pt_inc)
    denom = max_years_tp1 if max_years_tp1 > 0 else 1.0

    x_next = exp_years_next / denom
    return np.clip(x_next, 0.0, 1.0)


def _reference_solve(
    states,
    covariates,
    child_state_indexes,
    model_params,
    model_spec,
    prob_child,
    prob_partner,
):
    n_periods = model_spec.num_periods
    n_grid = model_spec.experience_grid_points
    exp_grid = np.linspace(0.0, 1.0, n_grid)

    n_states_per_period = int(states.shape[0] / n_periods)

    states_pp = states.reshape(n_periods, n_states_per_period, states.shape[1])
    covariates_pp = covariates.reshape(
        n_periods, n_states_per_period, covariates.shape[1]
    )

    child_idx_pp = child_state_indexes.reshape(n_periods, n_states_per_period, 3, 2, 2)

    # Match solve_python: indices are local to the next-period block.
    child_idx_local_pp = np.full_like(child_idx_pp, -999, dtype=int)
    for t in range(n_periods):
        if t < n_periods - 1:
            child_idx_local_pp[t] = child_idx_pp[t] - (t + 1) * n_states_per_period

    # Terminal emaxs_next: (n_states_per_period, n_grid, 4)
    emaxs_next = np.zeros((n_states_per_period, n_grid, 4))

    out_pp = np.zeros((n_periods, n_states_per_period, n_grid, 4))

    for t in reversed(range(n_periods)):
        states_t = states_pp[t]
        cov_t = covariates_pp[t]

        if t < n_periods - 1:
            child_local_t = child_idx_local_pp[t]
        else:
            child_local_t = None

        prob_child_t = prob_child[t]
        prob_partner_t = prob_partner[t]

        v_next_grid = emaxs_next[:, :, 3]

        emaxs_curr = np.zeros((n_states_per_period, n_grid, 4))

        for i in range(n_states_per_period):
            educ = int(states_t[i, EDUC_LEVEL])
            partner = int(states_t[i, PARTNER])

            pt_inc = float(
                get_pt_increment(
                    model_params=model_params,
                    educ_level=educ,
                    child_age=int(states_t[i, AGE_YOUNGEST_CHILD]),
                    biased_exp=False,
                )
            )

            p_child = float(prob_child_t[educ])
            p_partner = prob_partner_t[educ, partner, :].astype(float)

            # continuation_values[c, g]
            continuation_values = np.zeros((3, n_grid))

            for choice in range(3):
                if t == n_periods - 1:
                    continuation_values[choice] = 0.0
                    continue

                assert child_local_t is not None

                x_next = _next_stock_np(
                    x=exp_grid,
                    period=int(t),
                    init_exp_max=int(model_spec.init_exp_max),
                    pt_inc=pt_inc,
                    choice=choice,
                )

                idx00 = int(child_local_t[i, choice, 0, 0])
                idx01 = int(child_local_t[i, choice, 0, 1])
                idx10 = int(child_local_t[i, choice, 1, 0])
                idx11 = int(child_local_t[i, choice, 1, 1])

                v00 = np.interp(x_next, exp_grid, v_next_grid[idx00])
                v01 = np.interp(x_next, exp_grid, v_next_grid[idx01])
                v10 = np.interp(x_next, exp_grid, v_next_grid[idx10])
                v11 = np.interp(x_next, exp_grid, v_next_grid[idx11])

                no_child = 1.0 - p_child
                continuation_values[choice] = no_child * (
                    p_partner[0] * v00 + p_partner[1] * v01
                ) + p_child * (p_partner[0] * v10 + p_partner[1] * v11)

            max_years_t = int(model_spec.init_exp_max) + max(t, t * pt_inc)
            exp_years = exp_grid * max_years_t

            gamma_0 = float(model_params.gamma_0[educ])
            gamma_1 = float(model_params.gamma_1[educ])
            log_wage = gamma_0 + gamma_1 * np.log(exp_years + 1.0)

            female_wage_pt = HOURS[1] * np.exp(log_wage)
            female_wage_ft = HOURS[2] * np.exp(log_wage)

            # Taxes and transfers were set up so net income == gross income.
            equiv = float(cov_t[i, 2])
            cons_pt = np.maximum(female_wage_pt / equiv, 1e-14)
            cons_ft = np.maximum(female_wage_ft / equiv, 1e-14)

            # mu == 1 => utility == consumption
            val0 = model_params.delta * continuation_values[0]
            val1 = cons_pt + model_params.delta * continuation_values[1]
            val2 = cons_ft + model_params.delta * continuation_values[2]

            vmax = np.maximum(val0, np.maximum(val1, val2))

            emaxs_curr[i, :, 0] = continuation_values[0]
            emaxs_curr[i, :, 1] = continuation_values[1]
            emaxs_curr[i, :, 2] = continuation_values[2]
            emaxs_curr[i, :, 3] = vmax

        out_pp[t] = emaxs_curr
        emaxs_next = emaxs_curr

    return out_pp.reshape(-1, n_grid, 4)


def test_full_solve_matches_reference():
    model_spec = _make_min_model_spec()
    model_params = _make_min_model_params()

    # Exogenous processes.
    prob_child = np.full((model_spec.num_periods, model_spec.num_educ_levels), 0.3)

    prob_partner = np.zeros((model_spec.num_periods, model_spec.num_educ_levels, 2, 2))
    prob_partner[:, :, 0, 0] = 0.8
    prob_partner[:, :, 0, 1] = 0.2
    prob_partner[:, :, 1, 0] = 0.1
    prob_partner[:, :, 1, 1] = 0.9

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    _, emaxs = pyth_solve(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_params=model_params,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=False,
    )

    ref = _reference_solve(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_params=model_params,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
    )

    np.testing.assert_allclose(np.asarray(emaxs), ref, rtol=1e-10, atol=1e-10)
