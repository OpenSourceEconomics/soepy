import numpy as np
import pandas as pd

from soepy.shared.experience_stock import exp_years_to_stock
from soepy.shared.experience_stock import get_pt_increment
from soepy.shared.experience_stock import stock_to_exp_years
from soepy.shared.numerical_integration import draw_zero_one_distributed_shocks


def prepare_simulation_data(
    model_params,
    model_spec,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_pt,
    prob_exp_ft,
    is_expected,
):
    """Draw initial conditions and precompute grid objects for simulation."""

    initial_educ_level = np.random.choice(
        model_spec.num_educ_levels, model_spec.num_agents_sim, p=prob_educ_level
    )

    initial_period = np.asarray(model_spec.educ_years)[initial_educ_level]

    initial_child_age = np.full(model_spec.num_agents_sim, np.nan)
    initial_partner = np.full(model_spec.num_agents_sim, np.nan)
    initial_exp_pt = np.full(model_spec.num_agents_sim, np.nan)
    initial_exp_ft = np.full(model_spec.num_agents_sim, np.nan)

    for educ_level in range(model_spec.num_educ_levels):
        mask = initial_educ_level == educ_level

        initial_child_age[mask] = np.random.choice(
            list(range(-1, model_spec.child_age_init_max + 1)),
            mask.sum(),
            p=prob_child_age[educ_level],
        )

        initial_partner[mask] = np.random.binomial(
            size=mask.sum(),
            n=1,
            p=prob_partner_present[educ_level],
        )
        initial_exp_pt[mask] = np.random.choice(
            list(range(0, model_spec.init_exp_max + 1)),
            mask.sum(),
            p=prob_exp_pt[educ_level],
        )

        initial_exp_ft[mask] = np.random.choice(
            list(range(0, model_spec.init_exp_max + 1)),
            mask.sum(),
            p=prob_exp_ft[educ_level],
        )

    # Combine stocks
    pt_increment = get_pt_increment(
        model_params=model_params,
        educ_level=initial_educ_level,
        is_expected=is_expected,
    )

    total_years = initial_exp_pt * pt_increment + initial_exp_ft

    initial_exp_stock = exp_years_to_stock(
        exp_years=total_years,
        period=0,
        init_exp_max=model_spec.init_exp_max,
        pt_increment=pt_increment,
    )

    lagged_choice = lagged_choice_initial(initial_exp_years=total_years)

    unobserved_type = np.random.choice(
        np.arange(model_spec.num_types),
        model_spec.num_agents_sim,
        p=model_params.type_shares,
    )

    draws_sim = draw_zero_one_distributed_shocks(
        model_spec.seed_sim, model_spec.num_periods, model_spec.num_agents_sim
    )
    draws_sim = draws_sim * float(model_params.shock_sd)

    # prob_exp_pt/prob_exp_ft are passed in from simulate.py and originate from the
    # same legacy share files as prob_exp_years.
    initial_states = pd.DataFrame(
        {
            "Identifier": np.arange(model_spec.num_agents_sim, dtype=int),
            "Period": initial_period.astype(int),
            "Education_Level": initial_educ_level.astype(int),
            "Lagged_Choice": lagged_choice.astype(int),
            "Experience_Part_Time": initial_exp_pt.astype(int),
            "Experience_Full_Time": initial_exp_ft.astype(int),
            "Experience_Stock": initial_exp_stock.astype(float),
            "Type": unobserved_type.astype(int),
            "Age_Youngest_Child": initial_child_age.astype(int),
            "Partner_Indicator": initial_partner.astype(int),
        }
    )

    return initial_states, draws_sim


def lagged_choice_initial(initial_exp_years):
    """Determine initial lagged choice from total experience.

    Rule (per project decision):
    - lagged_choice = 2 (full-time) if initial experience years > 1
    - otherwise lagged_choice = 0
    """

    lagged_choice = np.zeros_like(initial_exp_years, dtype=int)
    lagged_choice[initial_exp_years > 1] = 2
    # Check if initial years is a float if so, assign 1 part-time
    int_exp = initial_exp_years.astype(int)
    is_float = np.abs(initial_exp_years - int_exp) > 1e-8
    lagged_choice[is_float] = 1
    return lagged_choice
