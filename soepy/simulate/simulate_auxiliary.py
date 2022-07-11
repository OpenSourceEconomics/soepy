import numpy as np
import pandas as pd

from soepy.shared.shared_auxiliary import calculate_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_constants import DATA_FORMATS_SIM
from soepy.shared.shared_constants import DATA_LABLES_SIM
from soepy.shared.shared_constants import HOURS


def pyth_simulate(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_employment_consumption_resources,
    child_age_update_rule,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_ft,
    prob_exp_pt,
    prob_child,
    prob_partner,
    is_expected,
):
    """Simulate agent experiences."""

    np.random.seed(model_spec.seed_sim)

    # Draw initial condition: education level
    initial_educ_level = np.random.choice(
        model_spec.num_educ_levels, model_spec.num_agents_sim, p=prob_educ_level
    )

    # Draw initial conditions: age of youngest child, partner status,
    # experience full-time and experience part-time
    initial_child_age = np.full(model_spec.num_agents_sim, np.nan)
    initial_partner_status = np.full(model_spec.num_agents_sim, np.nan)
    initial_pt_exp = np.full(model_spec.num_agents_sim, np.nan)
    initial_ft_exp = np.full(model_spec.num_agents_sim, np.nan)

    for educ_level in range(model_spec.num_educ_levels):
        # Child
        initial_child_age[initial_educ_level == educ_level] = np.random.choice(
            list(range(-1, model_spec.child_age_init_max + 1)),
            sum(initial_educ_level == educ_level),
            p=prob_child_age[educ_level],
        )
        # Partner
        initial_partner_status[initial_educ_level == educ_level] = np.random.binomial(
            size=sum(initial_educ_level == educ_level),
            n=1,
            p=prob_partner_present[educ_level],
        )

        # Part-time experience
        initial_pt_exp[initial_educ_level == educ_level] = np.random.choice(
            list(range(0, model_spec.init_exp_max + 1)),
            sum(initial_educ_level == educ_level),
            p=prob_exp_pt[educ_level],
        )
        # Full-time experience
        initial_ft_exp[initial_educ_level == educ_level] = np.random.choice(
            list(range(0, model_spec.init_exp_max + 1)),
            sum(initial_educ_level == educ_level),
            p=prob_exp_ft[educ_level],
        )
    # Draw random type
    type_ = np.random.choice(
        list(np.arange(model_spec.num_types)),
        model_spec.num_agents_sim,
        p=model_params.type_shares,
    )

    # Draw shocks
    attrs_spec = ["seed_sim", "num_periods", "num_agents_sim"]
    draws_sim = draw_disturbances(
        *[getattr(model_spec, attr) for attr in attrs_spec], model_params
    )

    # Calculate utility components
    log_wage_systematic, non_consumption_utilities = calculate_utility_components(
        model_params, model_spec, states, covariates, is_expected
    )

    # Determine initial states according to initial conditions
    initial_states = pd.DataFrame(
        np.column_stack(
            (
                np.arange(model_spec.num_agents_sim),
                np.array(model_spec.educ_years)[initial_educ_level],
                initial_educ_level,
                np.zeros(model_spec.num_agents_sim),
                initial_pt_exp,
                initial_ft_exp,
                type_,
                initial_child_age,
                initial_partner_status,
            )
        ),
        columns=DATA_LABLES_SIM[:9],
    ).astype(np.int)

    tax_splitting = model_spec.tax_splitting
    data = []

    # Loop over all periods
    for period in range(model_spec.num_periods):

        initial_states_in_period = initial_states.loc[
            initial_states.Period.eq(period)
        ].to_numpy()

        # Get all agents in the period.
        if period == 0:
            current_states = initial_states_in_period
        else:
            current_states = np.vstack((current_states, initial_states_in_period))

        idx = indexer[
            current_states[:, 1],  # 0 period
            current_states[:, 2],  # 1 educ_level
            current_states[:, 3],  # 2 lagged_choice
            current_states[:, 4],  # 3 exp_pt
            current_states[:, 5],  # 4 exp_ft
            current_states[:, 6],  # 5 type
            current_states[:, 7],  # 6 age_youngest_child
            current_states[:, 8],  # 7 partner_indicator
        ]

        # Extract corresponding utilities
        current_log_wage_systematic = log_wage_systematic[idx]
        current_non_consumption_utilities = non_consumption_utilities[idx]
        current_non_employment_consumption_resources = (
            non_employment_consumption_resources[idx]
        )
        current_equivalence_scale = covariates[idx][:, 2]
        current_male_wages = covariates[idx][:, 1]
        current_child_benefits = covariates[idx][:, 3]

        current_wages = np.exp(
            current_log_wage_systematic + draws_sim[period, current_states[:, 0]]
        )

        current_female_income = current_wages[:, np.newaxis] * HOURS[np.newaxis, 1:]

        current_employment_consumption_resources = (
            calculate_employment_consumption_resources(
                model_spec.ssc_deductions,
                model_spec.tax_params,
                current_female_income,
                current_male_wages,
                tax_splitting,
            )
        )

        current_employment_consumption_resources += current_child_benefits.reshape(
            -1, 1
        )

        child_care_costs = get_child_care_cost_for_choice(
            covariates[idx][:, 0].astype(float), model_spec.child_care_costs
        )

        current_employment_consumption_resources -= child_care_costs

        # Join alternative consumption resources. Ensure positivity.
        current_consumption_resources = np.hstack(
            (
                current_non_employment_consumption_resources.reshape(-1, 1),
                current_employment_consumption_resources,
            )
        ).clip(min=np.finfo(float).eps)

        # Calculate total values for all choices
        flow_utilities = (
            (current_consumption_resources / current_equivalence_scale.reshape(-1, 1))
            ** model_params.mu
            / model_params.mu
            * current_non_consumption_utilities
        )

        # Extract continuation values for all choices
        continuation_values = emaxs[idx, :3]

        value_functions = flow_utilities + model_params.delta * continuation_values

        # Determine choice as option with highest choice specific value function
        choice = np.argmax(value_functions, axis=1)

        child_current_age = current_states[:, 7]

        # Update child age
        # Modification for simulations with very few periods
        # where maximum childbearing age is not reached by the end of the model
        if period == model_spec.num_periods - 1:
            child_new_age = child_current_age

        # Periods where the probability to have a child is still positive
        elif period <= model_spec.last_child_bearing_period:
            # Update current states according to exogenous processes
            # Relate to child age updating
            kids_current_draw = np.random.binomial(
                size=current_states.shape[0],
                n=1,
                p=prob_child[period + 1, current_states[:, 2]],
            )

            # Convert to age of child according to age update rule
            child_new_age = np.where(
                kids_current_draw == 0, child_age_update_rule[idx], 0
            )
            # Periods where no new child can arrive
        else:
            child_new_age = child_age_update_rule[idx]

        # Update partner status according to random draw
        current_partner_status = current_states[:, 8]
        new_partner_status = np.full(current_states.shape[0], np.nan)

        # Get individuals without partner
        current_states_no_partner = current_states[current_states[:, 8] == 0]
        partner_arrival_current_draw = np.random.binomial(
            size=current_states_no_partner.shape[0],
            n=1,
            p=prob_partner[period, current_states_no_partner[:, 2], 0, 1],
        )
        new_partner_status[current_states[:, 8] == 0] = partner_arrival_current_draw

        # Get individuals with partner
        current_states_with_partner = current_states[current_states[:, 8] == 1]
        partner_separation_current_draw = np.random.binomial(
            size=current_states_with_partner.shape[0],
            n=1,
            p=prob_partner[period, current_states_with_partner[:, 2], 1, 0],
        )
        new_partner_status[current_states[:, 8] == 1] = (
            current_partner_status[current_states[:, 8] == 1]
            - partner_separation_current_draw
        )

        # Record period experiences
        rows = np.column_stack(
            (
                current_states.copy(),
                choice,
                current_log_wage_systematic,
                current_wages,
                current_non_consumption_utilities,
                flow_utilities,
                continuation_values,
                value_functions,
                current_male_wages,
            )
        )

        data.append(rows)

        # Update current states according to choice
        current_states[:, 1] += 1
        current_states[:, 3] = choice
        current_states[:, 4] = np.where(
            choice == 1, current_states[:, 4] + 1, current_states[:, 4]
        )
        current_states[:, 5] = np.where(
            choice == 2, current_states[:, 5] + 1, current_states[:, 5]
        )
        current_states[:, 7] = child_new_age
        current_states[:, 8] = new_partner_status

    dataset = pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM).astype(
        DATA_FORMATS_SIM
    )
    dataset.loc[dataset["Choice"] == 0, "Wage_Observed"] = np.nan

    return dataset


def get_child_care_cost_for_choice(child_bins, child_care_costs):
    child_bins[child_bins > 2] = 0
    child_costs = np.zeros((child_bins.shape[0], 2))
    for choice in range(2):
        for age_bin in range(1, 3):
            child_costs[child_bins == age_bin, choice] = child_care_costs[
                age_bin, choice
            ]
    return child_costs
