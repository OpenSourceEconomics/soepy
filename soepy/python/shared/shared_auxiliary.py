import sys

import math
import numpy as np

from soepy.python.shared.shared_constants import MISSING_FLOAT


def draw_disturbances(num_draws, shocks_cov, seed):
    """Creates desired number of draws of a multivariate standard normal distribution."""

    # Set seed
    np.random.seed(seed)

    # Input parameters of the distribution
    mean = [0, 0, 0]
    shocks_cov_matrix = [
        [shocks_cov[0], 0, 0],
        [0, shocks_cov[1], 0],
        [0, 0, shocks_cov[2]],
    ]

    # Create draws from the standard normal distribution
    draws = np.random.multivariate_normal(mean, shocks_cov_matrix, num_draws)

    # Return function output
    return draws


def calculate_utilities(model_params, educ_level, exp_p, exp_f, draws):
    """Calculate period/flow utilities for all choices given state, period, and shocks."""

    # Calculate wage net of period productivity shock
    wage_systematic = calculate_wage_systematic(educ_level, exp_p, exp_f, model_params)

    # Calculate period wages for the three choices includings chocks' realizations
    period_wages = calculate_period_wages(model_params, wage_systematic, draws)

    # Calculate 1st part of the period utilities related to consumption
    consumption_utilities = calculate_consumption_utilities(model_params, period_wages)

    # Calculate total period utilities by multiplying U(.) component
    utilities = calculate_total_utilities(model_params, consumption_utilities)

    # Return function output
    return utilities, consumption_utilities, period_wages, wage_systematic


def calculate_wage_systematic(educ_level, exp_p, exp_f, model_params):
    """Calculate systematic wages, i.e., wages net of shock, for specified state."""

    # Initialize container
    wage_systematic = np.nan

    # Construct wage components
    gamma_s0 = np.dot(educ_level, model_params.optim_paras[0:3])
    gamma_s1 = np.dot(educ_level, model_params.optim_paras[3:6])
    period_exp_sum = exp_p * np.dot(educ_level, model_params.optim_paras[6:9]) + exp_f
    depreciation = 1 - np.dot(educ_level, model_params.optim_paras[9:12])

    # Calculate wage in the given state
    period_exp_total = period_exp_sum * depreciation + 1
    returns_to_exp = gamma_s1 * period_exp_total
    wage_systematic = np.exp(gamma_s0) * returns_to_exp

    # Return function output
    return wage_systematic  # This is a scalar, equal for all choices


def calculate_period_wages(model_params, wage_systematic, draws):
    """Calculate period wages for each choice including choice
    and period specific productivty shock.
    """

    # Initialize container
    period_wages = np.tile(np.nan, model_params.num_choices)

    # Take the exponential of the disturbances
    exp_draws = np.exp(draws)

    # Calculate choice specific wages including productivity shock
    period_wages = wage_systematic * exp_draws

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
    consumption_utilities[0] = (
        model_params.benefits ** model_params.mu / model_params.mu
    )

    consumption_utilities[1] = (
        consumption_utilities[1] ** model_params.mu / model_params.mu
    )

    consumption_utilities[2] = (
        consumption_utilities[2] ** model_params.mu / model_params.mu
    )

    # Return function output
    return consumption_utilities


def calculate_total_utilities(model_params, consumption_utilities):
    """Calculate total period utilities for each of the choices."""

    # Initialize container for utilities at state space point and period
    total_utilities = np.tile(np.nan, model_params.num_choices)

    # Calculate U(.) for the three available choices
    U_ = np.array(
        [
            math.exp(0.00),
            math.exp(model_params.optim_paras[12]),
            math.exp(model_params.optim_paras[13]),
        ]
    )

    # Calculate utilities for the avaibale joices N, P, F
    total_utilities = consumption_utilities * U_

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
    continuation_values = np.tile(MISSING_FLOAT, model_params.num_choices)

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
        continuation_values = np.tile(0.0, model_params.num_choices)

    # Return function output
    return continuation_values
