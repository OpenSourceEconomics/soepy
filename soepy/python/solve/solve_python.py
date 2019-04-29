from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import draw_disturbances
from soepy.python.solve.solve_auxiliary import construct_covariates
from soepy.python.solve.solve_auxiliary import calculate_utilities
from soepy.python.solve.solve_auxiliary import pyth_backward_induction
from soepy.python.shared.shared_helpers import convert_state_space


def pyth_solve(model_params):
    """Solve the model by backward induction."""

    # Create all necessary grids and objects related to the state space
    states, _ = pyth_create_state_space(model_params)

    # Convert new to old state space objects
    state_space_args = convert_state_space(model_params, states)

    # Create objects that depend only on the state space
    covariates = construct_covariates(state_space_args)

    attrs = ["seed_emax", "shocks_cov", "num_periods", "num_draws_emax"]
    draws_emax = draw_disturbances(*[getattr(model_params, attr) for attr in attrs])

    flow_utilities, _, _, _ = calculate_utilities(
        model_params, state_space_args, covariates, draws_emax
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    periods_emax = pyth_backward_induction(
        model_params, state_space_args, covariates, flow_utilities, draws_emax
    )

    # Return function output
    return state_space_args, periods_emax, covariates
