from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import construct_covariates
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utilities
from soepy.python.solve.solve_auxiliary import pyth_backward_induction


def pyth_solve(model_params):
    """Solve the model by backward induction.

    The solution routine performs four key operations:
    - create all nodes (state space points) of the decision tree (state space)
    that the agents might possibly reach.
    - create covariates that depend on the state space components at every
    state space point.
    - calculate the instantaneous/flow utilities for each possible choice at every
    state space point
    - calculate the continuation values for each choice at every
    state space point.

    Parameters
    __________
    model_params : namedtuple
        Namedtuple containing all information relevant for running a simulation.
        Includes parameters, dimensions, information on initial conditions, etc.

    Returns
    _______
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.
    covariates : np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.
    emax_period : np.array
        Expected maximum value function of the current state space point.
    """

    # Create all necessary grids and objects related to the state space
    states, indexer = pyth_create_state_space(model_params)

    # Create objects that depend only on the state space
    covariates = construct_covariates(states)

    attrs = ["seed_emax", "shocks_cov", "num_periods", "num_draws_emax"]
    draws_emax = draw_disturbances(*[getattr(model_params, attr) for attr in attrs])

    flow_utilities, _, _, _ = calculate_utilities(
        model_params, states, covariates, draws_emax
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs = pyth_backward_induction(model_params, states, indexer, flow_utilities)

    # Return function output
    return states, indexer, covariates, emaxs
