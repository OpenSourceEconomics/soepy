import numpy as np

import numba

from soepy.python.shared.shared_constants import NUM_CHOICES


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


def calculate_utilities(model_params, states, covariates, draws):
    """Calculate period/flow utilities for all choices given state, period, and shocks.

    Parameters
    ----------
    model_params : namedtuple
        Contains all parameters of the model including information on dimensions
        (number of periods, agents, random draws, etc.) and coefficients to be
        estimated.
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    covariates: np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.
    draws : np.ndarray
        Array with dimension (num_periods, num_draws, NUM_CHOICES). The number of draws is
        equal to num_draws_emax when the function is called during model solution,
        and equal to num_agents_sim when called during the simulation routine.

    Returns
    -------
    wage_systematic : array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    period_wages : np.ndarray
        Array with shape (num_states, num_draws, NUM_CHOICES). Contains the wages for
        the period given agent's period choice and error term draw.
    consumption_utilities : np.ndarray
        Array with shape (num_states, num_draws, NUM_CHOICES) containing part
        of the utility related to consumption.
    flow_utilities : np.ndarray
        Array with dimensions (num_states, num_draws, NUM_CHOICES) containing total
        flow utility of each choice given error term draw at each state.

    """

    # Calculate wage net of period productivity shock
    wage_systematic = calculate_wage_systematic(model_params, states, covariates)

    # Calculate period wages for the three choices including shocks' realizations
    period_wages = calculate_period_wages(model_params, states, wage_systematic, draws)

    # Calculate 1st part of the period utilities related to consumption
    consumption_utilities = calculate_consumption_utilities(model_params, period_wages)

    # Calculate total period utilities by multiplying U(.) component
    flow_utilities = calculate_total_utilities(model_params, consumption_utilities)

    # Return function output
    return flow_utilities, consumption_utilities, period_wages, wage_systematic


def calculate_wage_systematic(model_params, states, covariates):
    """Calculate systematic wages, i.e., wages net of shock, for all states."""

    exp_p, exp_f = states[:, 3], states[:, 4]
    educ_level = covariates

    # Construct wage components
    gamma_0s = np.dot(educ_level, np.array(model_params.gamma_0s))
    gamma_1s = np.dot(educ_level, np.array(model_params.gamma_1s))
    period_exp_sum = exp_p * np.dot(educ_level, np.array(model_params.g_s)) + exp_f
    depreciation = 1 - np.dot(educ_level, np.array(model_params.delta_s))

    # Calculate wage in the given state
    period_exp_total = period_exp_sum * depreciation + 1
    returns_to_exp = np.exp(gamma_1s * np.log(period_exp_total))
    wage_systematic = np.exp(gamma_0s) * returns_to_exp

    # Return function output
    return wage_systematic


def calculate_period_wages(model_params, states, wage_systematic, draws):
    """Calculate period wages for each choice including choice
    and period specific productivity shock.
    """
    # Take the exponential of the disturbances
    exp_draws = np.exp(draws)

    shape = (wage_systematic.shape[0], draws.shape[1], NUM_CHOICES)
    period_wages = np.full(shape, np.nan)

    # Calculate choice specific wages including productivity shock
    for state in range(period_wages.shape[0]):
        period = states[state, 0]
        period_wages[state, :, :] = wage_systematic[state] * exp_draws[period, :, :]

    # Ensure that the benefits are recorded as non-labor income in data frame
    period_wages[:, :, 0] = model_params.benefits

    # Return function output
    return period_wages


@numba.jit(nopython=True)
def calculate_consumption_utilities(model_params, period_wages):
    """Calculate the first part of the period utilities related to consumption."""

    # Define hours array, possibly move to another file
    hours = np.array([0, 18, 38])

    # Calculate choice specific wages including productivity shock
    consumption_utilities = hours * period_wages
    consumption_utilities[:, :, 0] = model_params.benefits

    consumption_utilities = (
        np.power(consumption_utilities, model_params.mu) / model_params.mu
    )

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
