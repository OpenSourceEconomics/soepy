from soepy.solve.solve_auxiliary import pyth_create_state_space
from soepy.solve.solve_auxiliary import construct_covariates
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.solve.solve_auxiliary import pyth_backward_induction


def pyth_solve(model_params, model_spec, is_expected):
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
        Namedtuple containing all structural, potentially free and estimable,
        parameters relevant for running a simulation.
    model_spec : namedtuple
        Namedtuple containing all fixed parameters relevant for running a simulation
    is_expected: bool
        A boolean indicator that differentiates between the human capital accumulation
        process that agents expect (is_expected = True) and that the market generates
        (is_expected = False)

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
    emaxs : np.ndarray
        Array with shape (num states, num_choices +1). First block of dimension
        num_choices contains continuation values of the state space point.
        Lat element contains the expected maximum value function of the state space point.
    """

    # Create all necessary grids and objects related to the state space
    states, indexer = pyth_create_state_space(model_spec)

    # Create objects that depend only on the state space
    covariates = construct_covariates(states, model_spec)

    attrs_spec = ["seed_emax", "num_periods", "num_draws_emax"]
    draws_emax = draw_disturbances(
        *[getattr(model_spec, attr) for attr in attrs_spec], model_params
    )

    log_wage_systematic, non_consumption_utilities = calculate_utility_components(
        model_params, model_spec, states, covariates, is_expected
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs = pyth_backward_induction(
        model_spec,
        states,
        indexer,
        log_wage_systematic,
        non_consumption_utilities,
        draws_emax,
    )

    # Return function output
    return states, indexer, covariates, emaxs
