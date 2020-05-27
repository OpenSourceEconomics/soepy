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
    prob_partner,
    prob_educ_years,
    is_expected,
):
    """Simulate agent experiences."""

    # Draw random initial conditions
    educ_years = list(range(model_spec.educ_min, model_spec.educ_max + 1))
    np.random.seed(model_spec.seed_sim)
    initial_educ_years = np.random.choice(
        educ_years, model_spec.num_agents_sim, p=prob_educ_years
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
                initial_educ_years - model_spec.educ_min,
                initial_educ_years,
                np.zeros((model_spec.num_agents_sim, 3)),
                type_,
            )
        ),
        columns=DATA_LABLES_SIM[:7],
    ).astype(np.int)

    data = []

    # Loop over all periods
    for period in range(model_spec.num_periods):

        initial_states_in_period = initial_states.loc[
            initial_states.Years_of_Education.eq(period + model_spec.educ_min)
        ].to_numpy()

        # Draw indicator of child appearing in the first period
        kids_init_draw = np.random.binomial(
            size=initial_states_in_period.shape[0], n=1, p=prob_child[period],
        )

        # Convert to init age of child
        child_init_age = np.where(kids_init_draw == 0, -1, 0)

        # Draw presence of partner in the first period
        # Discuss: Best way to implement random draws by education
        partner_status_init_draw = np.random.binomial(
            size=initial_states_in_period.shape[0], n=1, p=0,
        )

        # Add columns to state space
        initial_states_in_period = np.c_[
            initial_states_in_period, child_init_age, partner_status_init_draw
        ]

        # Get all agents in the period.
        if period == 0:
            current_states = initial_states_in_period
        else:
            current_states = np.vstack((current_states, initial_states_in_period))

        idx = indexer[
            current_states[:, 1],
            current_states[:, 2] - model_spec.educ_min,
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
        #current_budget_constraint_components = np.full(current_budget_constraint_components.shape, 0.00)
        # debugging
        # print(current_budget_constraint_components)
        # end debugging
        current_non_consumption_utilities = non_consumption_utilities[idx]
        current_non_employment_benefits = non_employment_benefits[idx]

        # debugging
        current_wages_old = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        )
        current_wages_old[:, 0] = current_non_employment_benefits
        # end debugging

        current_wages = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        ) + current_budget_constraint_components.reshape(-1, 1)

        current_wages[:, 0] = (
            current_non_employment_benefits + current_budget_constraint_components
        )

        # debugging
        np.testing.assert_equal(current_wages_old, current_wages)
        #print("current wages old \n", current_wages_old)
        #print("current wages new \n", current_wages)
        # end debugging

        # Calculate total values for all choices
        flow_utilities = np.full((current_states.shape[0], 3), np.nan)

        # debugging
        flow_utilities_old = flow_utilities
        flow_utilities_old[:, :1] = (
            current_non_employment_benefits ** model_spec.mu / model_spec.mu
        ).reshape(current_states.shape[0], 1) * current_non_consumption_utilities[:, :1]
        flow_utilities_old[:, 1:] = (
            (HOURS[1:] * current_wages_old[:, 1:]) ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities[:, 1:]
        )
        # end debugging

        flow_utilities[:, :1] = (
            (current_non_employment_benefits + current_budget_constraint_components)
            ** model_spec.mu
            / model_spec.mu
        ).reshape(current_states.shape[0], 1) * current_non_consumption_utilities[:, :1]
        flow_utilities[:, 1:] = (
            (HOURS[1:] * current_wages[:, 1:]) ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities[:, 1:]
        )

        # debugging
        #print("flow_utilities_old \n", flow_utilities_old)
        #print("flow utilities \n", flow_utilities)
        np.testing.assert_equal(flow_utilities_old, flow_utilities)
        # end debugging

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
                size=current_states.shape[0], n=1, p=prob_child[period + 1],
            )

            # Convert to age of child according to age update rule
            child_current_age = np.where(
                kids_current_draw == 0, child_age_update_rule[idx], 0
            )
            # Periods where no new child can arrive
        else:
            child_current_age = child_age_update_rule[idx]

        # Update partner status according to random draw
        # Discuss: Best way to implement random draws by education
        # Applies here as well
        partner_current_draw = np.random.binomial(
            size=current_states.shape[0], n=1, p=0,
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
        current_states[:, 7] = child_current_age
        current_states[:, 8] = partner_current_draw

    dataset = pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM).astype(
        DATA_FORMATS_SIM
    )
    return dataset
