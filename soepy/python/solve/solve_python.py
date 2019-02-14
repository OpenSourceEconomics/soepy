from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import pyth_backward_induction


def pyth_solve(attr_dict):
    """Solve the model by backward induction."""

    # Create all necessary grids and objects related to the state space
    state_space_args = pyth_create_state_space(attr_dict)

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    periods_emax = pyth_backward_induction(attr_dict, state_space_args)

    # Return function output
    return state_space_args, periods_emax
