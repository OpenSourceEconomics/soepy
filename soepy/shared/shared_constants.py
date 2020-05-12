import numpy as np

# Set values of constants used across all modules here
MISSING_INT = -99
INVALID_FLOAT = -99.0
NUM_CHOICES = 3
IND = 2  # Stands for indicator, i.e., an indicator variable takes 2 values, 0 or 1
# Hours worked per month
# Assumption: weekly working hours times 4.5 weeks in a month
HOURS = np.array([0, 18 * 4.5, 38 * 4.5])
DATA_LABLES_SIM = [
    "Identifier",
    "Period",
    "Years_of_Education",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Type",
    "Age_Youngest_Child",
    "Partner_Indicator",
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

DATA_FORMATS_SIM = {
    key: (np.int if key in DATA_LABLES_SIM[:9] else np.float) for key in DATA_LABLES_SIM
}
