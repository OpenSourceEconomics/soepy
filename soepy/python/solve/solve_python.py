from soepy.python.solve.solve_auxiliary import pyth_create_state_space
from soepy.python.solve.solve_auxiliary import construct_covariates
from soepy.python.shared.shared_auxiliary import draw_disturbances
from soepy.python.shared.shared_auxiliary import calculate_utilities
from soepy.python.solve.solve_auxiliary import pyth_backward_induction


def pyth_solve(model_params):
    """Solve the model by backward induction."""

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
    periods_emax = pyth_backward_induction(
        model_params, states, indexer, covariates, flow_utilities
    )

    # Return function output
    return states, indexer, periods_emax, covariates
