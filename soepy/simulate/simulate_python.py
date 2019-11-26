from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.solve.solve_python import pyth_solve


def simulate(model_params_init_file_name, model_spec_init_file_name, is_expected=True):
    """Create a data frame of individuals' simulated experiences."""

    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)
    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Obtain model solution
    states, indexer, covariates, emaxs = pyth_solve(
        model_params, model_spec, is_expected
    )

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(
        model_params, model_spec, states, indexer, emaxs, covariates, is_expected=False
    )

    return df
