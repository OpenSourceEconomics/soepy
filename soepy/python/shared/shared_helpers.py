import numpy as np
from soepy.python.shared.shared_constants import MISSING_INT, NUM_CHOICES


def convert_state_space(model_params, states):
    """Converts new state space objects
    to the old state space objects and shapes.
    Allows for isolated testing of new state space.
    To be removed after speed up of soepy is complete."""

    # Define shapes
    shape = (
        model_params.num_periods,
        model_params.educ_range,
        NUM_CHOICES,
        model_params.num_periods,
        model_params.num_periods,
    )
    mapping_states_index_converted = np.tile(MISSING_INT, shape)

    num_states_period_upper_bound = np.prod(mapping_states_index_converted.shape)

    states_all_converted = np.tile(
        MISSING_INT, (model_params.num_periods, num_states_period_upper_bound, 4)
    )

    states_number_period_converted = np.tile(MISSING_INT, model_params.num_periods)

    # Create old state space objects from new ones
    for period in range(model_params.num_periods):

        # Extract states of the period
        states_period = states[states[:, 0] == period]

        # Determine number of states for the period
        states_number_period_converted[period] = states_period.shape[0]

        for k in range(states_number_period_converted[period]):
            current_state = states_period[k]
            _, educ, choice, exp_p, exp_f = current_state

            # Save state information in old format
            states_all_converted[period, k, :] = [educ, choice, exp_p, exp_f]

            # Record index in old format
            mapping_states_index_converted[period, educ - 10, choice, exp_p, exp_f] = k

    # Record max states period
    max_states_period_converted = max(states_number_period_converted)

    # Collect arguments
    state_space_args = (
        states_all_converted,
        states_number_period_converted,
        mapping_states_index_converted,
        max_states_period_converted,
    )

    # Return function output
    return state_space_args
