import numpy as np
import pandas as pd

from soepy.shared.shared_constants import HOURS, DATA_LABLES_SIM, DATA_FORMATS_SIM
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_auxiliary import calculate_utility_components


def pyth_simulate(
    model_params, model_spec, states, indexer, emaxs, covariates, is_expected
):
    """Simulate agent experiences."""

    # Draw random initial conditions
    educ_years = list(range(model_spec.educ_min, model_spec.educ_max + 1))
    np.random.seed(model_spec.seed_sim)
    initial_educ_years = np.random.choice(educ_years, model_spec.num_agents_sim)

    # Draw random type
    type_ = np.random.choice(
        list(np.arange(model_spec.num_types)),
        model_spec.num_agents_sim,
        p=model_params.type_shares,
    )

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
        ]

        # Extract corresponding utilities
        current_log_wage_systematic = log_wage_systematic[idx]
        current_non_consumption_utilities = non_consumption_utilities[idx]

        current_wages = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states[:, 0]]
        )
        current_wages[:, 0] = model_spec.benefits

        # Calculate total values for all choices
        flow_utilities = np.full((current_states.shape[0], 3), np.nan)

        flow_utilities[:, :1] = (
            model_spec.benefits ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities[:, :1]
        )
        flow_utilities[:, 1:] = (
            (HOURS[1:] * current_wages[:, 1:]) ** model_spec.mu
            / model_spec.mu
            * current_non_consumption_utilities[:, 1:]
        )

        # Extract continuation values for all choices
        continuation_values = emaxs[idx, :3]

        value_functions = flow_utilities + model_spec.delta * continuation_values

        # Determine choice as option with highest choice specific value function
        choice = np.argmax(value_functions, axis=1)

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

        # Update current states
        current_states[:, 1] += 1
        current_states[:, 3] = choice
        current_states[:, 4] = np.where(
            choice == 1, current_states[:, 4] + 1, current_states[:, 4]
        )
        current_states[:, 5] = np.where(
            choice == 2, current_states[:, 5] + 1, current_states[:, 5]
        )

    dataset = (
        pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM)
        .astype(DATA_FORMATS_SIM)
        .set_index(["Identifier", "Period"], drop=False)
    )

    # Fill gaps in history with NaNs.
    index = pd.MultiIndex.from_product(
        [range(model_spec.num_agents_sim), range(model_spec.num_periods)]
    )
    dataset = dataset.reindex(index)

    return dataset
