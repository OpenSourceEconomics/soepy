import numpy as np
import pandas as pd

from soepy.python.pre_processing.model_processing import read_init_file
from soepy.python.simulate.simulate_auxiliary import replace_missing_values
from soepy.python.simulate.simulate_auxiliary import pyth_simulate
from soepy.python.solve.solve_python import pyth_solve


def simulate(init_file_name):
    """Create a data frame of individuals' simulated experiences."""
    # Read in model specification from yaml file
    attr_dict = read_init_file(init_file_name)

    # Obtain model solution
    state_space_args, periods_emax = pyth_solve(attr_dict)

    # Simulate agents experiences according to parameters in the model specification
    dataset = pyth_simulate(attr_dict, state_space_args, periods_emax)

    # Create fixed objects needed to record simulated dataset to Pandas Dataframe
    # Define column lables
    DATA_LABLES_SIM = []
    DATA_LABLES_SIM += ["Identifier", "Period"]
    DATA_LABLES_SIM += ["Years of Education"]
    DATA_LABLES_SIM += ["Choice"]
    DATA_LABLES_SIM += ["Systematic Wage"]
    DATA_LABLES_SIM += ["Period Wage N", "Period Wage P", "Period Wage F"]
    DATA_LABLES_SIM += [
        "Consumption Utility N",
        "Consumption Utility P",
        "Consumption Utility F",
    ]
    DATA_LABLES_SIM += ["Flow Utility N", "Flow Utility P", "Flow Utility F"]

    # Define data types for data set columns
    DATA_FORMATS_SIM = dict()
    for key_ in DATA_LABLES_SIM:
        DATA_FORMATS_SIM[key_] = np.int
        if key_ in [
            "Choice",
            "Systematic Wage",
            "Period Wage N",
            "Period Wage P",
            "Period Wage F",
            "Consumption Utility N",
            "Consumption Utility P",
            "Consumption Utility F",
            "Flow Utility N",
            "Flow Utility P",
            "Flow Utility F",
        ]:
            DATA_FORMATS_SIM[key_] = np.float

    # Create data frame from simulated dataset
    data_frame = pd.DataFrame(
        data=replace_missing_values(dataset), columns=DATA_LABLES_SIM
    )

    # Set specific columns to desired data types
    data_frame = data_frame.astype(DATA_FORMATS_SIM)

    # Define identifier for unique observation in the data frame

    # Record function output
    return data_frame
