import numpy as np

# Set values of constants used across all modules here
MISSING_INT = -99
INVALID_FLOAT = -99.0
NUM_CHOICES = 3
HOURS = np.array([0, 18, 38])

DATA_LABLES_SIM = [
    "Identifier",
    "Period",
    "Years_of_Education",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Type",
    "Choice",
    "Log_Systematic_Wage",
    "Period_Wage_N",
    "Period_Wage_P",
    "Period_Wage_F",
    "Non_Consumption_Utility_N",
    "Non_Consumption_Utility_P",
    "Non_Consumption_Utility_F",
    "Flow_Utility_N",
    "Flow_Utility_P",
    "Flow_Utility_F",
    "Continuation_Value_N",
    "Continuation_Value_P",
    "Continuation_Value_F",
    "Value_Function_N",
    "Value_Function_P",
    "Value_Function_F",
]

# Define data types for data set columns
DATA_FORMATS_SIM = {
    key: (np.int if key in DATA_LABLES_SIM[:7] else np.float) for key in DATA_LABLES_SIM
}
