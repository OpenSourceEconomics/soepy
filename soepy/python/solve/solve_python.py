from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import pyth_backward_induction
from soepy.python.shared.shared_helpers import convert_state_space


def pyth_solve(model_params):
    """Solve the model by backward induction."""

    # Create all necessary grids and objects related to the state space
    states, _ = pyth_create_state_space(model_params)

    # Convert new to old state space objects
    state_space_args = convert_state_space(model_params, states)

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    periods_emax = pyth_backward_induction(model_params, state_space_args)

    # Return function output
    return state_space_args, periods_emax
