import numpy as np

from soepy.python.shared.shared_constants import NUM_COLUMNS_DATAFRAME
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utilities
from soepy.python.shared.shared_auxiliary import calculate_continuation_values


def pyth_simulate(model_params, states, indexer, periods_emax, covariates):
    """Simulate agent experiences."""

    # Draw random initial conditions
    educ_years = list(range(model_params.educ_min, model_params.educ_max + 1))
    np.random.seed(model_params.seed_sim)
    educ_years = np.random.choice(educ_years, model_params.num_agents_sim)

    attrs = ["seed_sim", "shocks_cov", "num_periods", "num_agents_sim"]
    draws_sim = draw_disturbances(*[getattr(model_params, attr) for attr in attrs])

    # Calculate utilities
    flow_utilities, cons_utilities, period_wages, wage_sys = calculate_utilities(
        model_params, states, covariates, draws_sim
    )

    # Start count over all simulations/rows (number of agents times number of periods)
    count = 0

    # Initialize container for the final output
    num_columns = (
        NUM_COLUMNS_DATAFRAME
    )  # count of the information units we wish to record

    dataset = np.full(
        (model_params.num_agents_sim * model_params.num_periods, num_columns), np.nan
    )

    # Loop over all agents
    for i in range(model_params.num_agents_sim):

        # Construct additional education information
        educ_years_i, educ_level, educ_years_idx = extract_individual_covariates(
            educ_years, model_params.educ_min, i
        )

        # Extract the indicator of the initial state for the individual
        # depending on the individuals initial condition
        initial_state_index = indexer[educ_years_idx, educ_years_idx, 0, 0, 0]

        # Assign the initial state as current state
        current_state = states[initial_state_index, :].copy()

        # Loop over all remaining
        for period in range(model_params.num_periods):

            # Record agent identifier, period number, and years of education
            dataset[count, :2] = i, period
            dataset[count, 2:3] = educ_years_i

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
            cuurent_flow_utilities = flow_utilities[current_state_index, i, :]
            cuurent_cons_utilities = cons_utilities[current_state_index, i, :]
            cuurent_period_wages = period_wages[current_state_index, i, :]
            cuurent_wage_sys = wage_sys[current_state_index]

            # Obtain continuation values for all choices
            continuation_values = calculate_continuation_values(
                model_params,
                indexer,
                period,
                periods_emax,
                educ_years_idx,
                exp_p,
                exp_f,
            )

            # Calculate total values for all choices
            value_functions = (
                cuurent_flow_utilities + model_params.delta * continuation_values
            )

            # Determine choice as option with highest choice specific value function
            max_idx = np.argmax(value_functions)

            # Record period experiences
            dataset[count, 3:4] = max_idx
            dataset[count, 4:5] = cuurent_wage_sys
            dataset[count, 5:8] = cuurent_period_wages[:]
            dataset[count, 8:11] = cuurent_cons_utilities[:]
            dataset[count, 11:14] = cuurent_flow_utilities[:]

            # Update state space component experience
            current_state[max_idx + 2] += 1

            # Update state space component choice_lagged
            current_state[2] = max_idx

            # Update simulation/row count
            count += 1

    # Return function output
    return dataset


def extract_individual_covariates(educ_years, educ_min, i):
    """Constructs additional covariates given agent indicator."""
    # Determine education level given number of years of education
    # Would it be more efficient to do this somewhere else?

    # Unpack state space components
    educ_years_i = educ_years[i]

    # Extract education information
    if educ_years_i == 10:
        educ_level = [1, 0, 0]

    elif educ_years_i == 11:
        educ_level = [0, 1, 0]

    elif educ_years_i == 12:
        educ_level = [0, 0, 1]

    else:
        raise ValueError("Invalid number of years of education")

    educ_years_idx = educ_years_i - educ_min

    # Return function output
    return educ_years_i, educ_level, educ_years_idx
