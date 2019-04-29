import numpy as np

from soepy.python.shared.shared_constants import NUM_CHOICES, MISSING_INT


def draw_disturbances(seed, shocks_cov, num_periods, num_draws):
    """Creates desired number of draws of a multivariate standard normal distribution."""

    # Set seed
    np.random.seed(seed)

    # Input parameters of the distribution
    mean = [0, 0, 0]
    shocks_cov_matrix = np.zeros((3, 3), float)
    np.fill_diagonal(shocks_cov_matrix, shocks_cov)

    # Create draws from the standard normal distribution
    draws = np.random.multivariate_normal(
        mean, shocks_cov_matrix, (num_periods, num_draws)
    )

    # Return function output
    return draws


def calculate_utilities(model_params, state_space_args, covariates, draws):
    """Calculate period/flow utilities for all choices given state, period, and shocks."""

    # Calculate wage net of period productivity shock
    wage_systematic = calculate_wage_systematic(
        model_params, state_space_args, covariates
    )

    # Calculate period wages for the three choices including shocks' realizations
    period_wages = calculate_period_wages(
        model_params, state_space_args, wage_systematic, draws
    )

    # Calculate 1st part of the period utilities related to consumption
    consumption_utilities = calculate_consumption_utilities(model_params, period_wages)

    # Calculate total period utilities by multiplying U(.) component
    flow_utilities = calculate_total_utilities(model_params, consumption_utilities)

    # Return function output
    return flow_utilities, consumption_utilities, period_wages, wage_systematic


def calculate_wage_systematic(model_params, state_space_args, covariates):
    """Calculate systematic wages, i.e., wages net of shock, for specified state."""

    states_all, _, _, _, = state_space_args
    exp_p, exp_f = states_all[:, :, 2], states_all[:, :, 3]
    educ_level = covariates

    # Construct wage components
    gamma_0s = np.dot(educ_level, model_params.gamma_0s)
    gamma_1s = np.dot(educ_level, model_params.gamma_1s)
    period_exp_sum = exp_p * np.dot(educ_level, model_params.g_s) + exp_f
    depreciation = 1 - np.dot(educ_level, model_params.delta_s)

    # Calculate wage in the given state
    period_exp_total = period_exp_sum * depreciation + 1
    returns_to_exp = gamma_1s * period_exp_total
    wage_systematic = np.exp(gamma_0s) * returns_to_exp

    # Return function output
    return wage_systematic  # This is a scalar, equal for all choices
    # Dimension of outpus (num_periods by num_states)


def calculate_period_wages(model_params, state_space_args, wage_systematic, draws):
    """Calculate period wages for each choice including choice
    and period specific productivity shock.
    """
    states_all, _, _, _, = state_space_args

    # Take the exponential of the disturbances
    exp_draws = np.exp(draws)

    shape = (states_all.shape[0], states_all.shape[1], draws.shape[1], NUM_CHOICES)
    period_wages = np.full(shape, MISSING_INT)

    # Calculate choice specific wages including productivity shock
    for period in range(period_wages.shape[0]):
        for state in range(period_wages.shape[1]):
            period_wages[period, state, :, :] = (
                wage_systematic[period, state] * exp_draws[period, :, :]
            )

    # Ensure that the benefits are recorded as non-labor income in data frame
    period_wages[:, :, :, 0] = model_params.benefits

    # Return function output
    return (
        period_wages
    )  # This is a vector, difference between choices comes from disturbance term.


def calculate_consumption_utilities(model_params, period_wages):
    """Calculate the first part of the period utilities related to consumption."""

    # Define hours array, possibly move to another file
    hours = np.array([0, 18, 38])

    # Calculate choice specific wages including productivity shock
    consumption_utilities = hours * period_wages

    consumption_utilities[
        consumption_utilities == 0
    ] = 10  # Some positive invalid value

    consumption_utilities[:, :, :, 0] = (
        model_params.benefits ** model_params.mu
    ) / model_params.mu

    consumption_utilities[:, :, :, 1] = (
        consumption_utilities[:, :, :, 1] ** model_params.mu
    ) / model_params.mu

    consumption_utilities[:, :, :, 2] = (
        consumption_utilities[:, :, :, 2] ** model_params.mu
    ) / model_params.mu

    # Return function output
    return consumption_utilities


def calculate_total_utilities(model_params, consumption_utilities):
    """Calculate total period utilities for each of the choices."""

    # Calculate U(.) for the three available choices
    u_ = np.array(
        [np.exp(0.00), np.exp(model_params.theta_p), np.exp(model_params.theta_f)]
    )

    # Calculate utilities for the available choices N, P, F
    total_utilities = consumption_utilities * u_

    # Return function output
    return total_utilities


def calculate_continuation_values(
    model_params,
    mapping_states_index,
    periods_emax,
    period,
    educ_years_idx,
    exp_p,
    exp_f,
):
    """Obtain continuation values for each of the choices."""

    # Initialize container for continuation values
    continuation_values = np.tile(np.nan, NUM_CHOICES)

    if period != (model_params.num_periods - 1):

        # Choice: Non-employment
        # Create index for extracting the continuation value
        future_idx = mapping_states_index[period + 1, educ_years_idx, 0, exp_p, exp_f]
        # Extract continuation value
        continuation_values[0] = periods_emax[period + 1, future_idx]

        # Choice: Part-time
        future_idx = mapping_states_index[
            period + 1, educ_years_idx, 1, exp_p + 1, exp_f
        ]
        continuation_values[1] = periods_emax[period + 1, future_idx]

        # Choice: Full-time
        future_idx = mapping_states_index[
            period + 1, educ_years_idx, 2, exp_p, exp_f + 1
        ]
        continuation_values[2] = periods_emax[period + 1, future_idx]

    else:
        continuation_values = np.tile(0.0, NUM_CHOICES)

    # Return function output
    return continuation_values
