import numpy as np
import pandas as pd

from soepy.python.shared.shared_constants import (
    HOURS,
    DATA_LABLES_SIM,
    DATA_FORMATS_SIM,
)
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utility_components


def pyth_simulate(model_params, states, indexer, emaxs, covariates):
    """Simulate agent experiences."""

    # Draw random initial conditions
    educ_years = list(range(model_params.educ_min, model_params.educ_max + 1))
    np.random.seed(model_params.seed_sim)
    initial_educ_years = np.random.choice(educ_years, model_params.num_agents_sim)

    attrs = ["seed_sim", "shocks_cov", "num_periods", "num_agents_sim"]
    draws_sim = draw_disturbances(*[getattr(model_params, attr) for attr in attrs])

    # Calculate utilities
    log_wage_systematic, nonconsumption_utilities = calculate_utility_components(
        model_params, states, covariates
    )

    # Create initial states.
    initial_states = pd.DataFrame(
        np.column_stack(
            (
                np.arange(model_params.num_agents_sim),
                initial_educ_years - model_params.educ_min,
                initial_educ_years,
                np.zeros((model_params.num_agents_sim, 3)),
            )
        ),
        columns=DATA_LABLES_SIM[:6],
    ).astype(np.int)

    data = []

    for period in range(model_params.num_periods):

        initial_states_in_period = initial_states.loc[
            initial_states.Years_of_Education.eq(period + model_params.educ_min)
        ].to_numpy()

        # Get all agents in the period.
        if period == 0:
            current_states = initial_states_in_period
        else:
            current_states = np.vstack((current_states, initial_states_in_period))

        idx = indexer[
            current_states[:, 1],
            current_states[:, 2] - model_params.educ_min,
            current_states[:, 3],
            current_states[:, 4],
            current_states[:, 5],
        ]

        # Extract corresponding utilities
        current_log_wage_systematic = log_wage_systematic[idx]

        current_wages = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        )
        current_wages[:, 0] = model_params.benefits

        # Extract continuation values for all choices
        continuation_values = emaxs[idx, :3]

        # Calculate total values for all choices
        flow_utilities = np.full((current_states.shape[0], 3), np.nan)

        flow_utilities[:, 0] = (
            model_params.benefits ** model_params.mu
            / model_params.mu
            * nonconsumption_utilities[0]
        )
        flow_utilities[:, 1:] = (
            (HOURS[1:] * current_wages[:, 1:]) ** model_params.mu
            / model_params.mu
            * nonconsumption_utilities[1:]
        )

        value_functions = flow_utilities + model_params.delta * continuation_values

        # Determine choice as option with highest choice specific value function
        choice = np.argmax(value_functions, axis=1)

        # Record period experiences
        rows = np.column_stack(
            (
                current_states.copy(),
                choice,
                current_log_wage_systematic,
                current_wages,
                np.tile(nonconsumption_utilities, (current_states.shape[0], 1)),
                continuation_values,
                value_functions,
            )
        )

        # Update current states
        current_states[:, 1] += 1
        current_states[:, 3] = choice
        # current_states[np.arange(current_states.shape[0]), choice + 3] = np.where(
        #     0 < choice,
        #     current_states[np.arange(current_states.shape[0]), choice + 3] + 1,
        #     current_states[np.arange(current_states.shape[0]), choice + 3],
        # )
        current_states[:, 4] = np.where(
            choice == 1, current_states[:, 4] + 1, current_states[:, 4]
        )
        current_states[:, 5] = np.where(
            choice == 2, current_states[:, 5] + 1, current_states[:, 5]
        )

        data.append(rows)

    dataset = (
        pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM)
        .astype(DATA_FORMATS_SIM)
        .set_index(["Identifier", "Period"], drop=False)
    )

    # Fill gaps in history with NaNs.
    index = pd.MultiIndex.from_product(
        [range(model_params.num_agents_sim), range(model_params.num_periods)]
    )
    dataset = dataset.reindex(index)

    return dataset
