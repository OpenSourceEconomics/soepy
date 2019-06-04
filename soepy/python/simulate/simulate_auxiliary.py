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

    current_states = pd.DataFrame(columns=DATA_LABLES_SIM).astype(int)

    data = []

    for period in range(model_params.num_periods):

        # Get all agents in the period.
        current_states = current_states.append(
            initial_states.loc[
                initial_states.Years_of_Education.eq(period + model_params.educ_min)
            ],
            sort=False,
        )

        idx = indexer[
            current_states.Period.to_numpy(),
            current_states.Years_of_Education.to_numpy() - model_params.educ_min,
            current_states.Lagged_Choice.to_numpy(),
            current_states.Experience_Part_Time.to_numpy(),
            current_states.Experience_Full_Time.to_numpy(),
        ]

        # Extract corresponding utilities
        current_log_wage_systematic = log_wage_systematic[idx]

        current_wages = np.exp(
            current_log_wage_systematic.reshape(-1, 1)
            + draws_sim[period, current_states.Identifier]
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
        period_df = current_states.copy()
        period_df["Choice"] = choice
        period_df["Log_Systematic_Wage"] = current_log_wage_systematic
        period_df[["Period_Wage_N", "Period_Wage_P", "Period_Wage_F"]] = current_wages
        period_df[
            [
                "Non_Consumption_Utility_N",
                "Non_Consumption_Utility_P",
                "Non_Consumption_Utility_F",
            ]
        ] = nonconsumption_utilities
        period_df[
            ["Continuation_Value_N", "Continuation_Value_P", "Continuation_Value_F"]
        ] = continuation_values
        period_df[
            ["Value_Function_N", "Value_Function_P", "Value_Function_F"]
        ] = value_functions

        # Update current states
        current_states.Period += 1
        current_states.Lagged_Choice = period_df.Choice
        current_states.loc[
            current_states.Lagged_Choice.eq(1), "Experience_Part_Time"
        ] += 1
        current_states.loc[
            current_states.Lagged_Choice.eq(2), "Experience_Full_Time"
        ] += 1

        data.append(period_df)

    dataset = (
        pd.concat(data, sort=False)
        .astype(DATA_FORMATS_SIM)
        .set_index(["Identifier", "Period"], drop=False)
    )

    # Fill gaps in history with NaNs.
    index = pd.MultiIndex.from_product(
        [range(model_params.num_agents_sim), range(model_params.num_periods)]
    )
    dataset = dataset.reindex(index)

    return dataset
