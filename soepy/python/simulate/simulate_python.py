from soepy.python.pre_processing.model_processing import transform_old_init_dict_to_df
from soepy.python.pre_processing.model_processing import read_model_params_init
from soepy.python.pre_processing.model_processing import read_model_spec_init
from soepy.python.simulate.simulate_auxiliary import pyth_simulate
from soepy.python.solve.solve_python import pyth_solve


def simulate(init_file_name):
    """Create a data frame of individuals' simulated experiences."""

    # Read in model specification from yaml file
    model_params_df = transform_old_init_dict_to_df(init_file_name)
    model_params = read_model_params_init(model_params_df)
    model_spec = read_model_spec_init(init_file_name)

    # Obtain model solution
    states, indexer, covariates, emaxs = pyth_solve(model_params, model_spec)

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(model_params, states, indexer, emaxs, covariates)

    return df
