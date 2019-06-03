import numpy as np

from soepy.python.shared.shared_constants import NUM_COLUMNS_DATAFRAME, HOURS
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utility_components


def pyth_simulate(model_params, states, indexer, emaxs, covariates):
    """Simulate agent experiences."""

    # Draw random initial conditions
    educ_years = list(range(model_params.educ_min, model_params.educ_max + 1))
    np.random.seed(model_params.seed_sim)
    educ_years = np.random.choice(educ_years, model_params.num_agents_sim)

    attrs = ["seed_sim", "shocks_cov", "num_periods", "num_agents_sim"]
    draws_sim = draw_disturbances(*[getattr(model_params, attr) for attr in attrs])

    # Calculate utilities
    log_wage_systematic, nonconsumption_utilities = calculate_utility_components(
        model_params, states, covariates
    )

    # Start count over all simulations/rows (number of agents times number of periods)
    count = 0

    dataset = np.full(
        (model_params.num_agents_sim * model_params.num_periods, NUM_COLUMNS_DATAFRAME),
        np.nan,
    )

    # Loop over all agents
    for i in range(model_params.num_agents_sim):

        # Construct additional education information
        educ_years_i = educ_years[i]
        educ_years_idx = educ_years_i - model_params.educ_min

        # Extract the indicator of the initial state for the individual
        # depending on the individuals initial condition
        initial_state_index = indexer[educ_years_idx, educ_years_idx, 0, 0, 0]

        # Assign the initial state as current state
        current_state = states[initial_state_index, :].copy()

        # Loop over all remaining
        for period in range(model_params.num_periods):

            # Record agent identifier, period number, and years of education
            dataset[count, :3] = i, period, educ_years_i

            # Make sure that experiences are recorded only after
            # the individual has completed education and entered the model
            if period < educ_years_idx:

                # Update count
                count += 1

                # Skip recording experiences and leave NaN in data set
                continue

            # Extract state space point index
            _, _, choice_lagged, exp_p, exp_f = current_state

            current_state_index = indexer[
                period, educ_years_idx, choice_lagged, exp_p, exp_f
            ]

            # Extract corresponding utilities
            current_log_wage_systematic = log_wage_systematic[current_state_index]

            current_wages = np.exp(
                current_log_wage_systematic + draws_sim[period, i]
            )
            current_wages[0] = model_params.benefits

            # Extract continuation values for all choices
            continuation_values = emaxs[current_state_index, :3]

            # Calculate total values for all choices
            flow_utilities = (
                (HOURS * current_wages) ** model_params.mu
                / model_params.mu
                * nonconsumption_utilities
            )

            value_functions = flow_utilities + model_params.delta * continuation_values

            # Determine choice as option with highest choice specific value function
            choice = np.argmax(value_functions)

            # Record period experiences
            dataset[count, 3] = choice
            dataset[count, 4] = current_log_wage_systematic
            dataset[count, 5:8] = current_wages
            dataset[count, 8:11] = nonconsumption_utilities
            dataset[count, 11:14] = continuation_values
            dataset[count, 14:17] = value_functions

            # Update state space component experience
            current_state[choice + 2] += 1

            # Update state space component choice_lagged
            current_state[2] = choice

            # Update simulation/row count
            count += 1

    return dataset
