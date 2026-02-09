import numpy as np
import pandas as pd

from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_component_vector
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.experience_stock import get_pt_increment


def _lagged_choice_initial(initial_exp_years):
    lagged_choice = np.zeros_like(initial_exp_years, dtype=int)
    lagged_choice[initial_exp_years > 1] = 2
    int_exp = initial_exp_years.astype(int)
    is_float = np.abs(initial_exp_years - int_exp) > 1e-8
    lagged_choice[is_float] = 1
    return lagged_choice


def create_initial_states_from_probs(
    model_params,
    model_spec,
    *,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_pt,
    prob_exp_ft,
):
    np.random.seed(model_spec.seed_sim)

    initial_educ_level = np.random.choice(
        model_spec.num_educ_levels,
        model_spec.num_agents_sim,
        p=prob_educ_level,
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

    pt_increment = get_pt_increment(
        model_params=model_params,
        educ_level=initial_educ_level,
        child_age=initial_child_age,
        biased_exp=False,
    )
    total_years = initial_exp_pt * pt_increment + initial_exp_ft
    lagged_choice = _lagged_choice_initial(total_years)

    unobserved_type = np.random.choice(
        np.arange(model_spec.num_types),
        model_spec.num_agents_sim,
        p=model_params.type_shares,
    )

    initial_states = pd.DataFrame(
        {
            "Identifier": np.arange(model_spec.num_agents_sim, dtype=int),
            "Period": initial_period.astype(int),
            "Education_Level": initial_educ_level.astype(int),
            "Lagged_Choice": lagged_choice.astype(int),
            "Experience_Part_Time": initial_exp_pt.astype(int),
            "Experience_Full_Time": initial_exp_ft.astype(int),
            "Type": unobserved_type.astype(int),
            "Age_Youngest_Child": initial_child_age.astype(int),
            "Partner_Indicator": initial_partner.astype(int),
        }
    )

    return initial_states


def create_initial_states(
    *,
    model_params_init_file_name,
    model_spec_init_file_name,
):
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)
    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    prob_educ_level = gen_prob_educ_level_vector(model_spec)
    prob_child_age = gen_prob_child_init_age_vector(model_spec)
    prob_partner_present = gen_prob_partner_present_vector(model_spec)
    prob_exp_pt = gen_prob_init_exp_component_vector(
        model_spec,
        model_spec.pt_exp_shares_file_name,
    )
    prob_exp_ft = gen_prob_init_exp_component_vector(
        model_spec,
        model_spec.ft_exp_shares_file_name,
    )

    return create_initial_states_from_probs(
        model_params=model_params,
        model_spec=model_spec,
        prob_educ_level=prob_educ_level,
        prob_child_age=prob_child_age,
        prob_partner_present=prob_partner_present,
        prob_exp_pt=prob_exp_pt,
        prob_exp_ft=prob_exp_ft,
    )
