import numpy as np
import pandas as pd

from soepy.shared.shared_constants import (
    HOURS,
    DATA_LABLES_SIM,
    DATA_FORMATS_SIM,
)
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_auxiliary import calculate_utility_components


def pyth_simulate(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    budget_constraint_components,
    non_employment_benefits,
    child_age_update_rule,
    prob_child,
    prob_partner_arrival,
    prob_educ_level,
    prob_child_age,
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
        # child
        initial_child_age[initial_educ_level == educ_level] = np.random.choice(
            list(range(-1, model_spec.child_age_init_max + 1)),
            sum(initial_educ_level == educ_level),
            p=prob_child_age[educ_level],
        )
        # Partner
        initial_partner_status[initial_educ_level == educ_level] = np.random.binomial(
            size=sum(initial_educ_level == educ_level),
            n=1,
            p=prob_partner_arrival[0, educ_level],
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
        current_budget_constraint_components = budget_constraint_components[idx]
        current_non_consumption_utilities = non_consumption_utilities[idx]
        current_non_employment_benefits = non_employment_benefits[idx]
        current_equivalence_scale = covariates[idx][:, 2]

        current_wages = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        )

        # Calculate total values for all choices
        flow_utilities = np.full((current_states.shape[0], 3), np.nan)

        flow_utilities[:, :1] = (
            (
                (current_non_employment_benefits + current_budget_constraint_components)
                / current_equivalence_scale
            )
            ** model_spec.mu
            / model_spec.mu
        ).reshape(current_states.shape[0], 1) * current_non_consumption_utilities[:, :1]

        flow_utilities[:, 1:] = (
            (
                (
                    HOURS[1:] * current_wages
                    + current_budget_constraint_components.reshape(-1, 1)
                )
                / current_equivalence_scale.reshape(current_states.shape[0], 1)
            )
            ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities[:, 1:]
        )

        # Extract continuation values for all choices
        continuation_values = emaxs[idx, :3]

        value_functions = flow_utilities + model_spec.delta * continuation_values

        # Determine choice as option with highest choice specific value function
        choice = np.argmax(value_functions, axis=1)

        # Modification for simulations with very few periods
        # where maximum childbearing age is not reached by the end of the model
        if period == model_spec.num_periods - 1:
            child_current_age = current_states[:, 7]
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
            child_current_age = np.where(
                kids_current_draw == 0, child_age_update_rule[idx], 0
            )
            # Periods where no new child can arrive
        else:
            child_current_age = child_age_update_rule[idx]

        # Update partner status according to random draw
        # Get individuals without partner
        current_states_no_partner = current_states[np.where(current_states[:, 8] == 0)]
        partner_current_draw = np.random.binomial(
            size=current_states_no_partner.shape[0],
            n=1,
            p=prob_partner_arrival[period, current_states_no_partner[:, 2]],
        )
        current_partner_status = current_states[:, 8]
        current_partner_status[
            np.where(current_states[:, 8] == 0)
        ] = partner_current_draw

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
        current_states[:, 7] = child_current_age
        current_states[:, 8] = current_partner_status

    dataset = pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM).astype(
        DATA_FORMATS_SIM
    )

    # Determine the period wage given choice in the period
    dataset["Wage_Observed"] = 0
    dataset.loc[dataset["Choice"] == 1, "Wage_Observed"] = dataset.loc[
        dataset["Choice"] == 1, "Period_Wage_P"
    ]
    dataset.loc[dataset["Choice"] == 2, "Wage_Observed"] = dataset.loc[
        dataset["Choice"] == 2, "Period_Wage_F"
    ]

    return dataset
