from soepy.python.pre_processing.model_processing import read_init_file
from soepy.python.simulate.simulate_auxiliary import pyth_simulate
from soepy.python.solve.solve_python import pyth_solve


def simulate(init_file_name):
    """Create a data frame of individuals' simulated experiences."""
    # Read in model specification from yaml file
    model_params = read_init_file(init_file_name)

    # Obtain model solution
    states, indexer, covariates, emaxs = pyth_solve(model_params)

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(model_params, states, indexer, emaxs, covariates)

    return df
