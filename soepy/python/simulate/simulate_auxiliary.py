import numpy as np

from .soepy.shared.shared_constants import MISSING_FLOAT

from soepy.shared.shared_auxiliary import draw_disturbances
from .soepy.shared.shared_auxiliary import calculate_wage_systematic
from .soepy.shared.shared_auxiliary import calculate_period_wages
from .soepy.shared.shared_auxiliary import calculate_consumption_utilities
from .soepy.shared.shared_auxiliary import calculate_total_utilities
from .soepy.shared.shared_auxiliary import calculate_utilities
from .soepy.shared.shared_auxiliary import calculate_continuation_values


def pyth_simulate(attr_dict, state_space_args, periods_emax):
    """Simulate agent experiences."""
    
    # Unpack objects from agrs
    states_all, states_number_period, mapping_states_index, max_states_period = state_space_args[0], state_space_args[1], state_space_args[2], state_space_args[3]
    
    # Unpack parameter from the model specification
    educ_min = attr_dict['INITIAL_CONDITIONS']['educ_min']
    educ_max = attr_dict['INITIAL_CONDITIONS']['educ_max']
    num_periods = attr_dict['GENERAL']['num_periods']
    num_agents_sim = attr_dict['SIMULATION']['num_agents_sim']
    seed_sim = attr_dict['SIMULATION']['seed_sim']
    shocks_cov = attr_dict['DERIVED_ATTR']['shocks_cov']
    optim_paras = attr_dict['PARAMETERS']['optim_paras']
    delta = attr_dict['CONSTANTS']['delta']

    educ_years = list(range(educ_min, educ_max + 1))
    educ_years = np.random.choice(educ_years, num_agents_sim)

    # Create draws for simulated sample
    draws_sim = draw_disturbances((num_periods, num_agents_sim), shocks_cov, seed_sim)

    # Start count over all simulations/row (number of agents times number of periods)
    count = 0

    # Initialize container for the final output
    num_columns = 14 # count of the information units we wish to record
    dataset = np.tile(MISSING_FLOAT, (num_agents_sim*num_periods, num_columns))

    # Loop over all agents
    for i in range(num_agents_sim):


        # Construct additional education information
        educ_years_i, educ_level, educ_years_idx = extract_individual_covariates (educ_years, educ_min, i)

        # Extract the indicator of the initial state for the individual
        # depending on the individuals initial condition
        initial_state_index = mapping_states_index[educ_years_idx, educ_years_idx, 0, 0, 0]

        # Assign the initial state as current state
        current_state = states_all[educ_years_idx, initial_state_index, :].copy()

        # Loop over all remaining
        for period in range(num_periods):

            # Extract state space components
            choice_lagged, exp_p, exp_f = current_state[1], current_state[2], current_state[3]

            # Look up the indicator for the current state
            k = mapping_states_index[period, educ_years_i - educ_min, choice_lagged, exp_p, exp_f]

            # Record agent identifier and current period number in the dataset
            dataset[count, :2] = i, period

            # Calculate choice specific value functions
            # for individual, period and state space point

            # Extract the error term draws corresponding to
            # period number and individual
            corresponding_draws = draws_sim[period, i, :]

            # Calculate correspongind flow utilities
            flow_utilities, consumption_utilities, period_wages, wage_systematic = calculate_utilities(attr_dict,
                                                                                                       educ_level,
                                                                                                       exp_p,
                                                                                                       exp_f,
                                                                                                       optim_paras,
                                                                                                       corresponding_draws)

            # Obtain continuation values for all choices
            continuation_values = calculate_continuation_values(attr_dict,
                                                                mapping_states_index,
                                                                periods_emax,
                                                                period,
                                                                educ_years_idx,
                                                                exp_p,
                                                                exp_f)

            # Calculate total values for all choices
            value_functions = flow_utilities + delta * continuation_values

            # Determine choice as option with highest choice specific value function
            max_idx = np.argmax(value_functions)


            # Record output
            # Record agent identifier, period number, and choice
            dataset[count, :2] = i, period, 
            dataset[count, 2:3] = educ_years_i
            dataset[count, 3:4] = max_idx
            dataset[count, 4:5] = wage_systematic
            dataset[count, 5:8] = period_wages[:]
            dataset[count, 8:11] = consumption_utilities[:]
            dataset[count, 11:14] = flow_utilities[:]


            # Update state space component experience
            current_state[max_idx + 1] += 1

            # Update state space component choice_lagged
            current_state[1] = max_idx

            # Update simulation/row count
            count += 1
    
    # Return function output
    return dataset


def extract_individual_covariates (educ_years, educ_min, i):
    """Constructs additional covariates given agent indicator."""
    
    # Determine education level given number of years of education
    # Would it be more efficient to do this somewhere else?

    # Unpack state space components
    educ_years_i = educ_years[i]

    # Extract education information
    if (educ_years_i <= 10):
        educ_level = [1,0,0]

    elif (educ_years_i > 10) and (educ_years_i <= 12):
        educ_level = [0,1,0]

    else:
        educ_level = [0,0,1]

    educ_years_idx = educ_years_i - educ_min
    
    # Return function output
    return educ_years_i, educ_level, educ_years_idx


def replace_missing_values (arguments):
    """Replace MISSING_FLOAT with NAN."""
    
    # Antibugging
    assert isinstance(arguments, tuple) or isinstance(arguments, np.ndarray)

    if isinstance(arguments, np.ndarray):
        arguments = (arguments,)

    rslt = tuple()

    for argument in arguments:
        
        # Transform to float array to evaluate missing values
        argument_internal = np.asfarray(argument)

        # Determine missing values
        is_missing = argument_internal == MISSING_FLOAT
        if np.any(is_missing):
            # Replace missing values
            argument = np.asfarray(argument)
            argument[is_missing] = np.nan

        rslt += (argument,)

    # Align interface
    if len(rslt) == 1:
        rslt = rslt[0]

    # Function output
    return rslt