import numpy as np
import numba
import pandas as pd

from soepy.python.shared.shared_constants import (
    MISSING_INT,
    NUM_CHOICES,
    INVALID_FLOAT,
    HOURS,
)


def construct_covariates(states):
    """Construct a matrix of all the covariates
    that depend only on the state space.

    Parameters
    ---------
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of education,
        the lagged choice, years of experience in part-time and in full-time
         employment of the agent.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.

    """
    educ_level = pd.Series(states[:, 1])
    covariates = pd.cut(educ_level, bins=[0, 10, 11, 12], labels=[0, 1, 2]).to_numpy()

    return covariates


@numba.jit(nopython=True)
def pyth_create_state_space(model_params):
    """Create state space object.

    The state space consists of all admissible combinations of the following components:
    period, years of education, lagged choice, full-time experience (F),
    and part-time experience (P).

    :data:`states` stores the information on states in a tabular format.
    Each row of the table corresponds to one admissible state space point
    and contains the values of the state space components listed above.
    :data:`indexer` is a multidimensional array where each component
    of the state space corresponds to one dimension. The values of the array cells
    index the corresponding state space point in :data:`states`.
    Traversing the state space requires incrementing the indices of :data:`indexer`
    and selecting the corresponding state space point component values in :data:`states`.

    Parameters
    ----------
    model_params: namedtuple
        Namedtuple containing all information relevant for running a simulation.
        Includes parameters, dimensions, information on initial conditions, etc.

    Returns
    -------
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.

    Examples
    --------
    >>> from collections import namedtuple
    >>> model_params = namedtuple("model_params", "num_periods educ_range educ_min")
    >>> model_params = model_params(10, 3, 10)
    >>> NUM_CHOICES = 3
    >>> states, indexer = pyth_create_state_space(
    ...     model_params
    ... )
    >>> states.shape
    (1110, 5)
    >>> indexer.shape
    (10, 3, 3, 10, 10)
    """
    data = []

    # Array for mapping the state space points (states) to indices
    shape = (
        model_params.num_periods,
        model_params.educ_range,
        NUM_CHOICES,
        model_params.num_periods,
        model_params.num_periods,
    )

    indexer = np.full(shape, MISSING_INT)

    # Initialize counter for admissible state space points
    i = 0

    # Loop over all periods / all ages
    for period in range(model_params.num_periods):

        # Loop over all possible initial conditions for education
        for educ_years in range(model_params.educ_range):

            # Check if individual has already completed education
            # and will make a labor supply choice in the period
            if educ_years > period:
                continue

            # Loop over all admissible years of experience accumulated in full-time
            for exp_f in range(model_params.num_periods):

                # Loop over all admissible years of experience accumulated in part-time
                for exp_p in range(model_params.num_periods):

                    # The accumulation of experience cannot exceed time elapsed
                    # since individual entered the model
                    if exp_f + exp_p > period - educ_years:
                        continue

                    # Add an additional entry state
                    # [educ_years + model_params.educ_min, 0, 0, 0]
                    # for individuals who have just completed education
                    # and still have no experience in any occupation.
                    if period == educ_years:

                        # Assign an additional integer count i
                        # for entry state
                        indexer[period, educ_years, 0, 0, 0] = i

                        # Record the values of the state space components
                        # for the currently reached entry state
                        row = [period, educ_years + model_params.educ_min, 0, 0, 0]

                        # Update count once more
                        i += 1

                        data.append(row)

                    else:

                        # Loop over the three labor market choices, N, P, F
                        for choice_lagged in range(NUM_CHOICES):

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

                            # If an individual has always been employed,
                            # she cannot have non-employment (0) as lagged choice
                            if (choice_lagged == 0) and (
                                exp_f + exp_p == period - educ_years
                            ):
                                continue

                            # Check for duplicate states
                            if (
                                indexer[period, educ_years, choice_lagged, exp_p, exp_f]
                                != MISSING_INT
                            ):
                                continue

                            # Assign the integer count i as an indicator for the
                            # currently reached admissible state space point
                            indexer[period, educ_years, choice_lagged, exp_p, exp_f] = i

                            # Update count
                            i += 1

                            # Record the values of the state space components
                            # for the currently reached admissible state space point
                            row = [
                                period,
                                educ_years + model_params.educ_min,
                                choice_lagged,
                                exp_p,
                                exp_f,
                            ]

                            data.append(row)

        states = np.array(data)

    # Return function output
    return states, indexer


def pyth_backward_induction(
    model_params, states, indexer, log_wage_systematic, nonconsumption_utilities, draws
):
    """Get expected maximum value function at every state space point.

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
    indexer : np.ndarray
        Array where each dimension represents a componenet of the state space.
        :data:`states[k]` returns the values of the state space components
        at state :data:`k`. Indexing :data:`indexer` by the same state space
        component values returns :data:`k`.
    flow_utilities : np.ndarray
        Array with dimensions (num_states, num_draws, NUM_CHOICES) containing total
        flow utility of each choice given error term draw at each state.

    Returns
    -------
    emaxs : np.ndarray
        An array of dimension (num_states, num choices + 1). The object's rows contain
        the continuation values of each choice at the specific state space points
        as its first elements. The last row element corresponds to the maximum
        expected value function of the state.
    """

    emaxs = np.zeros((states.shape[0], NUM_CHOICES + 1))

    # Loop backwards over all periods
    for period in reversed(range(model_params.num_periods)):

        # Extract period information
        states_period = states[np.where(states[:, 0] == period)]  # as slice
        log_wage_systematic_period = log_wage_systematic[states[:, 0] == period]

        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == model_params.num_periods - 1:
            pass
        else:

            # Fill first block of elements in emaxs for the current period
            # corresponding to the continuation values
            emaxs = get_continuation_values(model_params, states_period, indexer, emaxs)

        # Extract current period information for current loop calculation
        emaxs_period = emaxs[np.where(states[:, 0] == period)]

        # Calculate emax for current period reached by the loop
        emax_period = construct_emax(
            model_params.delta,
            log_wage_systematic_period,
            nonconsumption_utilities,
            draws[period],
            emaxs_period[:, :3],
            HOURS,
            model_params.mu,
            model_params.benefits,
        )
        emaxs_period[:, 3] = emax_period
        emaxs[np.where(states[:, 0] == period)] = emaxs_period

    return emaxs


@numba.njit(nogil=True)
def get_continuation_values(model_params, states_subset, indexer, emaxs):
    """Obtain continuation values for each of the choices at each state
    of the period currently reached by the parent loop.

    This function takes a parent node and looks up the continuation values
    of each of the available choices. It takes the entire block of
    data:`emaxs` corresponding to the period and fills in the first block
    of elements corresponding to the continuation values.
    The continuation value of each choice is the expected maximum value
    function of the next period's state if the particular choice was
    taken this period. The expected maximum value function values are
    contained as the last element of the data:`emaxs` row of next
    period's state.

    Warning
    -------
    This function must be extremely performant as the lookup is done for each state in a
    state space (except for states in the last period) for each evaluation of the
    optimization of parameters.
    """
    for i in range(states_subset.shape[0]):
        # Unpack parent state and get index
        period, educ_years, choice_lagged, exp_p, exp_f = states_subset[i]
        k_parent = indexer[
            period, educ_years - model_params.educ_min, choice_lagged, exp_p, exp_f
        ]

        # Choice: Non-employment
        k = indexer[period + 1, educ_years - model_params.educ_min, 0, exp_p, exp_f]
        emaxs[k_parent, 0] = emaxs[k, 3]

        # Choice: Part-time
        k = indexer[period + 1, educ_years - model_params.educ_min, 1, exp_p + 1, exp_f]
        emaxs[k_parent, 1] = emaxs[k, 3]

        # Choice: Full-time
        k = indexer[period + 1, educ_years - model_params.educ_min, 2, exp_p, exp_f + 1]
        emaxs[k_parent, 2] = emaxs[k, 3]

    return emaxs


@numba.guvectorize(
    ["f8, f8, f8[:], f8[:, :], f8[:], f8[:], f8, f8, f8[:]"],
    "(), (), (n_choices), (n_draws, n_choices), (n_choices), (n_choices), (), () -> ()",
    nopython=True,
    target="parallel",
)
def construct_emax(
    delta,
    log_wage_systematic,
    nonconsumption_utilities,
    draws,
    emaxs,
    hours,
    mu,
    benefits,
    emax,
):
    """Simulate expected maximum utility for a given distribution of the unobservables.

    The function calculates the maximum expected value function over the distribution of
    the error term at each state space point in the period currently reached by the
    parent loop. The expectation calculation is performed via `Monte Carlo
    integration`_. The goal is to approximate an integral by evaluating the integrand at
    randomly chosen points. In this setting, one wants to approximate the expected
    maximum utility of the current state.

    Parameters
    ----------
    delta : int
        Dynamic discount factor.
    flow_utilities_period : np.ndarray
        Array with dimensions (number of states in period, num_draws, NUM_CHOICES)
        containing total flow utility of each choice given error term draw
        at each state.
    emaxs : np.ndarray
        An array of dimension (num. states in period, num choices + 1).
        The object's rows contain the continuation values of each choice at the specific
        state space points as its first elements. The last row element corresponds
        to the maximum expected value function of the state. This column is
        full of zeros for the input object.

    Returns
    -------
    emax : np.array
        Expected maximum value function of the current state space point.
        Array of lentgh number of states in the current period. The vector
        corresponds to the second block of values in the data:`emaxs` object.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws, num_choices = draws.shape

    emax[0] = 0.0

    for i in range(num_draws):

        current_max_total_utility = INVALID_FLOAT

        for j in range(num_choices):

            wage = np.exp(log_wage_systematic + draws[i, j])

            if j == 0:
                consumption_utility = benefits ** mu / mu
            else:
                consumption_utility = (hours[j] * wage) ** mu / mu

            total_utility = consumption_utility * nonconsumption_utilities[j]

            if total_utility > current_max_total_utility:
                current_max_total_utility = total_utility

        emax[0] += current_max_total_utility

    emax[0] /= num_draws
