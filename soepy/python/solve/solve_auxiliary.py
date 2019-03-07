import numpy as np

from soepy.python.shared.shared_constants import MISSING_INT, MISSING_FLOAT
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utilities
from soepy.python.shared.shared_auxiliary import calculate_continuation_values


def pyth_create_state_space(attr_dict):
    """Create state space related objects
    given state space components in model specification.
    """

    # Unpack parameter from the model specification
    num_choices = attr_dict["GENERAL"]["num_choices"]
    num_periods = attr_dict["GENERAL"]["num_periods"]
    educ_min = attr_dict["INITIAL_CONDITIONS"]["educ_min"]
    educ_range = attr_dict["DERIVED_ATTR"]["educ_range"]

    # Array for mapping the state space points (states) to indices
    shape = (num_periods, educ_range, num_choices, num_periods, num_periods)
    mapping_states_index = np.tile(MISSING_INT, shape)

    # Maximum number of state space points per period. There
    # can be no more states in a period than this number.
    num_states_period_upper_bound = np.prod(mapping_states_index.shape)

    # Array to collect all state space points that can be reached each period
    states_all = np.tile(MISSING_INT, (num_periods, num_states_period_upper_bound, 4))

    # Array for the maximum number state space points per period
    states_number_period = np.tile(MISSING_INT, num_periods)

    # Loop over all periods / all ages
    for period in range(num_periods):

        # Start count for admissible state space points
        k = 0

        # Loop over all possible initial conditions for education
        for educ_years in range(educ_range):

            # Check if individual has already completed education
            # and will make a labor supply choice in the period
            if educ_years > period:
                continue

            # Loop over all admissible years of experience accumulated in part-time
            for exp_f in range(num_periods):

                # Loop over all admaissible years of experience accumulated in full-time
                for exp_p in range(num_periods):

                    # The accumulation of experience cannot exceed time elapsed
                    # since individual entered the model
                    if exp_f + exp_p > period - educ_years:
                        continue

                    # Add an additional entry state [educ_years + educ_min, 0, 0, 0]
                    # for individuals who have just completed education
                    # and still have no experience in any occupation.
                    if period == educ_years:

                        # Assign an additional the integer count k
                        # for entry state
                        mapping_states_index[period, educ_years, 0, 0, 0] = k

                        # Record the values of the state space components
                        # for the currentry reached entry state
                        states_all[period, k, :] = [educ_years + educ_min, 0, 0, 0]

                        # Update count once more
                        k += 1

                    else:

                        # Loop over the three labor market choices, N, P, F
                        for choice_lagged in range(num_choices):

                            # If individual has only worked full-time in the past,
                            # she can only have full-time (2) as lagged choice
                            if (choice_lagged != 2) and (exp_f == period - educ_years):
                                continue

                            # If individual has only worked part-time in the past,
                            # she can only have part-time (1) as lagged choice
                            if (choice_lagged != 1) and (exp_p == period - educ_years):
                                continue

                            # If an individual has never worked full-time,
                            # she cannot have that lagged activity
                            if (choice_lagged == 2) and (exp_f == 0):
                                continue

                            # If an individual has never worked part-time,
                            # she cannot have that lagged activity
                            if (choice_lagged == 1) and (exp_p == 0):
                                continue

                            # Check for duplicate states
                            if (
                                mapping_states_index[
                                    period, educ_years, choice_lagged, exp_p, exp_f
                                ]
                                != MISSING_INT
                            ):
                                continue

                            # Assign the integer count k as an indicator for the
                            # currently reached admissible state space point
                            mapping_states_index[
                                period, educ_years, choice_lagged, exp_p, exp_f
                            ] = k

                            # Record the values of the state space components
                            # for the currently reached admissible state space point
                            states_all[period, k, :] = [
                                educ_years + educ_min,
                                choice_lagged,
                                exp_p,
                                exp_f,
                            ]

                            # Update count
                            k += 1

        # Record number of admissible state space points for the period currently
        # reached in the loop
        states_number_period[period] = k

    # Auxiliary objects
    max_states_period = max(states_number_period)

    # Collect arguments
    state_space_args = (
        states_all,
        states_number_period,
        mapping_states_index,
        max_states_period,
    )

    # Return function output
    return state_space_args


def pyth_backward_induction(attr_dict, state_space_args):
    """Obtain the value function maximum values
    for all admissible states and periods in a backward induction procedure.
    """

    # Unpack objects from agrs
    states_all, states_number_period, mapping_states_index, max_states_period = (
        state_space_args[0],
        state_space_args[1],
        state_space_args[2],
        state_space_args[3],
    )

    # Unpack parameter from the model specification
    num_periods = attr_dict["GENERAL"]["num_periods"]
    num_draws_emax = attr_dict["SOLUTION"]["num_draws_emax"]
    seed_emax = attr_dict["SOLUTION"]["seed_emax"]
    shocks_cov = attr_dict["DERIVED_ATTR"]["shocks_cov"]
    optim_paras = attr_dict["PARAMETERS"]["optim_paras"]

    # Initialize container for the final result,
    # maximal value function per perdiod and state:
    periods_emax = np.tile(MISSING_FLOAT, (num_periods, max_states_period))

    draws_emax = draw_disturbances((num_periods, num_draws_emax), shocks_cov, seed_emax)

    # Construct covariates
    covariates = construct_covariates(
        states_all, states_number_period, max_states_period, attr_dict
    )

    # Loop over all periods
    for period in range(num_periods - 1, -1, -1):

        # Select the random draws for Monte Carlo integration relevant for the period
        draws_emax_period = draws_emax[period, :, :]

        # Loop over all admissible state space points
        # for the period currently reached by the parent loop
        for k in range(states_number_period[period]):

            # Construct additional education information
            educ_level, educ_years_idx = covariates[period, k, 0:3], covariates[period, k, 3]

            # Integrate out the error term
            emax = construct_emax(
                attr_dict,
                period,
                k,
                educ_level,
                educ_years_idx,
                num_draws_emax,
                draws_emax_period,
                states_all,
                mapping_states_index,
                optim_paras,
                periods_emax,
            )

            # Record function output
            periods_emax[period, k] = emax

    # Return function output
    return periods_emax


def construct_covariates(
    states_all, states_number_period, max_states_period, attr_dict
):
    """Constructs additional covariates given state space components."""

    # Unpack attributes from the model specification:
    num_periods = attr_dict["GENERAL"]["num_periods"]
    educ_min = attr_dict["INITIAL_CONDITIONS"]["educ_min"]

    # Initialize covariates array
    covariates = np.tile(MISSING_INT, (num_periods, max_states_period, 4))

    # Fill in education information
    for period in range(num_periods):

        for k in range(states_number_period[period]):

            educ_years = states_all[period, k, 0]

            # Extract education information
            if educ_years <= 10:
                covariates[period, k, 0:3] = [1, 0, 0]

            elif (educ_years > 10) and (educ_years <= 12):
                covariates[period, k, 0:3] = [0, 1, 0]

            elif educ_years > 12:
                covariates[period, k, 0:3] = [0, 0, 1]

            else:
                raise ValueError("Invalid number of years of education")

            covariates[period, k, 3] = educ_years - educ_min

    # Return final output
    return covariates


def construct_emax(
    attr_dict,
    period,
    k,
    educ_level,
    educ_years_idx,
    num_draws_emax,
    draws_emax_period,
    states_all,
    mapping_states_index,
    optim_paras,
    periods_emax,
):
    """Integrate out the error terms in a Monte Carlo simulation procedure
    to obtain value function maximum values for each period and state.
    """

    # Unpack attributes from the model specification
    delta = attr_dict["CONSTANTS"]["delta"]

    # Initialize container for sum of value function maximum values
    # over all error term draws for the period and state
    emax = 0.0

    # Loop over all error term draws
    # for the period and state currently rached by the parent loop
    for i in range(num_draws_emax):

        # Extract the error term draws corresponding to
        # period number, state, and loop iteration number, i
        corresponding_draws = draws_emax_period[i, :]

        # Extract relevant state space components
        educ_years, _, exp_p, exp_f = states_all[period, k, :]

        # Calculate flow utility at current period, state, and draw
        flow_utilities = calculate_utilities(
            attr_dict, educ_level, exp_p, exp_f, optim_paras, corresponding_draws
        )[0]

        # Obtain continuation values for all choices
        continuation_values = calculate_continuation_values(
            attr_dict,
            mapping_states_index,
            periods_emax,
            period,
            educ_years_idx,
            exp_p,
            exp_f,
        )

        # Calculate choice specific value functions
        value_functions = flow_utilities + delta * continuation_values

        # Obtain highest value function value among the available choices. If above
        # draws were the true shocks, maximum is the the current period value function
        # value. It is the sum the flow utility and next periods value function given an
        # optimal decision in the future and an optimal choice in the current period.
        maximum = max(value_functions)

        # Add to sum over all draws
        emax += maximum

        # End loop

    # Average over the number of draws
    emax = emax / num_draws_emax

    # Thus, we have integrated out the error term

    # Return function output
    return emax
