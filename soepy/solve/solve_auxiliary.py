import numpy as np
import numba
import pandas as pd

from soepy.shared.shared_constants import MISSING_INT, NUM_CHOICES, INVALID_FLOAT, HOURS


def construct_covariates(states, model_spec):
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
    covariates = pd.cut(
        educ_level,
        bins=[0, model_spec.low_bound, model_spec.middle_bound, model_spec.high_bound],
        labels=[0, 1, 2],
    ).to_numpy()

    return covariates


@numba.jit(nopython=True)
def pyth_create_state_space(model_spec):
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
    model_spec: namedtuple
        Namedtuple containing all fixed parameters describing the model and its
         state space that are relevant for running a simulation.

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
    >>> model_spec = namedtuple(
    ...     "model_specification", "num_periods educ_range educ_min num_types"
    ... )
    >>> model_spec = model_spec(10, 3, 10, 2)
    >>> NUM_CHOICES = 3
    >>> states, indexer = pyth_create_state_space(
    ...     model_spec
    ... )
    >>> states.shape
    (2220, 6)
    >>> indexer.shape
    (10, 3, 3, 10, 10, 2)
    """
    data = []

    # Array for mapping the state space points (states) to indices
    shape = (
        model_spec.num_periods,
        model_spec.educ_range,
        NUM_CHOICES,
        model_spec.num_periods,
        model_spec.num_periods,
        model_spec.num_types,
    )

    indexer = np.full(shape, MISSING_INT)

    # Initialize counter for admissible state space points
    i = 0

    # Loop over all periods / all ages
    for period in range(model_spec.num_periods):

        for type_ in range(model_spec.num_types):

            # Loop over all possible initial conditions for education
            for educ_years in range(model_spec.educ_range):

                # Check if individual has already completed education
                # and will make a labor supply choice in the period
                if educ_years > period:
                    continue

                # Loop over all admissible years of experience accumulated in full-time
                for exp_f in range(model_spec.num_periods):

                    # Loop over all admissible years of experience accumulated
                    # in part-time
                    for exp_p in range(model_spec.num_periods):

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
                            indexer[period, educ_years, 0, 0, 0, type_] = i

                            # Record the values of the state space components
                            # for the currently reached entry state
                            row = [
                                period,
                                educ_years + model_spec.educ_min,
                                0,
                                0,
                                0,
                                type_,
                            ]

                            # Update count once more
                            i += 1

                            data.append(row)

                        else:

                            # Loop over the three labor market choices, N, P, F
                            for choice_lagged in range(NUM_CHOICES):

                                # If individual has only worked full-time in the past,
                                # she can only have full-time (2) as lagged choice
                                if (choice_lagged != 2) and (
                                    exp_f == period - educ_years
                                ):
                                    continue

                                # If individual has only worked part-time in the past,
                                # she can only have part-time (1) as lagged choice
                                if (choice_lagged != 1) and (
                                    exp_p == period - educ_years
                                ):
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
                                    indexer[
                                        period,
                                        educ_years,
                                        choice_lagged,
                                        exp_p,
                                        exp_f,
                                        type_,
                                    ]
                                    != MISSING_INT
                                ):
                                    continue

                                # Assign the integer count i as an indicator for the
                                # currently reached admissible state space point
                                indexer[
                                    period,
                                    educ_years,
                                    choice_lagged,
                                    exp_p,
                                    exp_f,
                                    type_,
                                ] = i

                                # Update count
                                i += 1

                                # Record the values of the state space components
                                # for the currently reached admissible state space point
                                row = [
                                    period,
                                    educ_years + model_spec.educ_min,
                                    choice_lagged,
                                    exp_p,
                                    exp_f,
                                    type_,
                                ]

                                data.append(row)

        states = np.array(data)

    # Return function output
    return states, indexer


def pyth_backward_induction(
    model_spec, states, indexer, log_wage_systematic, non_consumption_utilities, draws
):
    """Get expected maximum value function at every state space point.
    Backward induction is performed all at once for all states in a given period.
    The function loops through each period. The included construct_emax function
    implicitly loops through all states in the period currently reached by the
    parent loop.

    Parameters
    ----------
    model_spec : namedtuple
        Contains all fixed parameters of the model including information on dimensions
        such as number of periods, agents, random draws, etc.
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        Array where each dimension represents a componenet of the state space.
        :data:`states[k]` returns the values of the state space components
        at state :data:`k`. Indexing :data:`indexer` by the same state space
        component values returns :data:`k`.
    log_wage_systematic : np.array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.

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
    for period in reversed(range(model_spec.num_periods)):

        # Extract period information
        states_period = states[np.where(states[:, 0] == period)]
        log_wage_systematic_period = log_wage_systematic[states[:, 0] == period]
        non_consumption_utilities_period = non_consumption_utilities[
            states[:, 0] == period
        ]

        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == model_spec.num_periods - 1:
            pass
        else:

            # Fill first block of elements in emaxs for the current period
            # corresponding to the continuation values
            emaxs = get_continuation_values(model_spec, states_period, indexer, emaxs)

        # Extract current period information for current loop calculation
        emaxs_period = emaxs[np.where(states[:, 0] == period)]

        # Calculate emax for current period reached by the loop
        emax_period = construct_emax(
            model_spec.delta,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            draws[period],
            emaxs_period[:, :3],
            HOURS,
            model_spec.mu,
            model_spec.benefits,
        )
        emaxs_period[:, 3] = emax_period
        emaxs[np.where(states[:, 0] == period)] = emaxs_period

    return emaxs


@numba.njit(nogil=True)
def get_continuation_values(model_spec, states_subset, indexer, emaxs):
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
        period, educ_years, choice_lagged, exp_p, exp_f, type_ = states_subset[i]
        k_parent = indexer[
            period, educ_years - model_spec.educ_min, choice_lagged, exp_p, exp_f, type_
        ]

        # Choice: Non-employment
        k = indexer[
            period + 1, educ_years - model_spec.educ_min, 0, exp_p, exp_f, type_
        ]
        emaxs[k_parent, 0] = emaxs[k, 3]

        # Choice: Part-time
        k = indexer[
            period + 1, educ_years - model_spec.educ_min, 1, exp_p + 1, exp_f, type_
        ]
        emaxs[k_parent, 1] = emaxs[k, 3]

        # Choice: Full-time
        k = indexer[
            period + 1, educ_years - model_spec.educ_min, 2, exp_p, exp_f + 1, type_
        ]
        emaxs[k_parent, 2] = emaxs[k, 3]

    return emaxs


@numba.njit
def _get_max_aggregated_utilities(
    delta,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    emaxs,
    hours,
    mu,
    benefits,
):

    current_max_value_function = INVALID_FLOAT

    for j in range(NUM_CHOICES):

        wage = np.exp(log_wage_systematic + draws[j])

        if j == 0:
            consumption_utility = benefits ** mu / mu
        else:
            consumption_utility = (hours[j] * wage) ** mu / mu

        value_function_choice = (
            consumption_utility * non_consumption_utilities[j] + delta * emaxs[j]
        )

        if value_function_choice > current_max_value_function:
            current_max_value_function = value_function_choice

    return current_max_value_function


@numba.guvectorize(
    ["f8, f8, f8[:], f8[:, :], f8[:], f8[:], f8, f8, f8[:]"],
    "(), (), (n_choices), (n_draws, n_choices), (n_choices), (n_choices), (), () -> ()",
    nopython=True,
    target="parallel",
)
def construct_emax(
    delta,
    log_wage_systematic,
    non_consumption_utilities,
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
    integration`. The goal is to approximate an integral by evaluating the integrand at
    randomly chosen points. In this setting, one wants to approximate the expected
    maximum utility of a given state.

    Parameters
    ----------
    delta : int
        Dynamic discount factor.
    log_wage_systematic : array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.
    draws : np.ndarray
        Array of dimension (num_periods, num_choices, num_draws). Randomly drawn
        realisations of the error term used to integrate out the distribution of
        the error term.
    emaxs : np.ndarray
        An array of dimension (num. states in period, num choices + 1).
        The object's rows contain the continuation values of each choice at the specific
        state space points as its first elements. The last row element corresponds
        to the maximum expected value function of the state. This column is
        full of zeros for the input object.
    hours : np.array
        Array of constants, corresponding to the working hours associated with
        each employment choice.
    mu : int
        Constant governing the degree of risk aversion and inter-temporal
        substitution in the model.
    benefits : int
        Constant level of hourly income received in case of choice N,
        non-employment.

    Returns
    -------
    emax : np.array
        Expected maximum value function of the current state space point.
        Array of lentgh number of states in the current period. The vector
        corresponds to the second block of values in the data:`emaxs` object.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws = draws.shape[0]

    emax[0] = 0.0

    for i in range(num_draws):

        max_total_utility = _get_max_aggregated_utilities(
            delta,
            log_wage_systematic,
            non_consumption_utilities,
            draws[i],
            emaxs,
            hours,
            mu,
            benefits,
        )

        emax[0] += max_total_utility

    emax[0] /= num_draws
