import numpy as np
import pandas as pd

from soepy.shared.shared_constants import (
    HOURS,
    DATA_LABLES_SIM,
    DATA_FORMATS_SIM,
)
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.shared.shared_auxiliary import calculate_employment_consumption_resources


def pyth_simulate(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    child_age_update_rule,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_child,
    prob_partner_arrival,
    prob_partner_separation,
    is_expected,
):
    """Simulate agent experiences."""

    np.random.seed(model_spec.seed_sim)

    # Draw initial condition: education level
    initial_educ_level = np.random.choice(
        model_spec.num_educ_levels, model_spec.num_agents_sim, p=prob_educ_level
    )

    # Draw initial conditions: age of youngest child and partner status
    initial_child_age = np.full(model_spec.num_agents_sim, np.nan)
    initial_partner_status = np.full(model_spec.num_agents_sim, np.nan)

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
                np.zeros((model_spec.num_agents_sim, 3)),
                type_,
                initial_child_age,
                initial_partner_status,
            )
        ),
        columns=DATA_LABLES_SIM[:9],
    ).astype(np.int)

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
            current_states[:, 1],
            current_states[:, 2],
            current_states[:, 3],
            current_states[:, 4],
            current_states[:, 5],
            current_states[:, 6],
            current_states[:, 7],
            current_states[:, 8],
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
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        )

        current_hh_income = HOURS[1:] * current_wages + current_male_wages.reshape(
            -1, 1
        )

        current_hh_income = current_hh_income.reshape(
            2 * current_wages.shape[0], order="F"
        )

        current_employment_consumption_resources = (
            calculate_employment_consumption_resources(
                deductions_spec,
                income_tax_spec,
                current_hh_income,
            )
        )

        current_employment_consumption_resources = (
            current_employment_consumption_resources.reshape(
                current_wages.shape[0], 2, order="F"
            )
            + current_child_benefits.reshape(-1, 1)
        )

        current_consumption_ressources = np.hstack(
            (
                current_non_employment_consumption_resources.reshape(-1, 1),
                current_employment_consumption_resources,
            )
        )

        # Calculate total values for all choices
        flow_utilities = (
            (
                (current_consumption_ressources)
                / current_equivalence_scale.reshape(-1, 1)
            )
            ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities
        )

        # Extract continuation values for all choices
        continuation_values = emaxs[idx, :3]

        value_functions = flow_utilities + model_spec.delta * continuation_values

        # Determine choice as option with highest choice specific value function
        choice = np.argmax(value_functions, axis=1)

        child_current_age = current_states[:, 7]
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
                p=prob_child[period + 1],
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
            p=prob_partner_arrival[period, current_states_no_partner[:, 2]],
        )
        new_partner_status[current_states[:, 8] == 0] = partner_arrival_current_draw

        # Get individuals with partner
        current_states_with_partner = current_states[current_states[:, 8] == 1]
        partner_separation_current_draw = np.random.binomial(
            size=current_states_with_partner.shape[0],
            n=1,
            p=prob_partner_separation[period, current_states_with_partner[:, 2]],
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

    # Determine the period wage given choice in the period
    dataset["Wage_Observed"] = np.nan
    dataset.loc[dataset["Choice"] == 1, "Wage_Observed"] = dataset.loc[
        dataset["Choice"] == 1, "Period_Wage_P"
    ]
    dataset.loc[dataset["Choice"] == 2, "Wage_Observed"] = dataset.loc[
        dataset["Choice"] == 2, "Period_Wage_F"
    ]

    return dataset
