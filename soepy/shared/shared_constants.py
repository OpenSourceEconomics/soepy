import numpy as np

# Set values of constants used across all modules here
MISSING_INT = -99
INVALID_FLOAT = -99.0
NUM_CHOICES = 3
# Hours worked per month
# Assumption: weekly working hours times 4.5 weeks in a month
HOURS = np.array([0, 18, 38])
DATA_LABLES_SIM = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Type",
    "Age_Youngest_Child",
    "Partner_Indicator",
    "Choice",
    "Log_Systematic_Wage",
    "Wage_Observed",
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
    "Male_Wages",
]
DATA_FORMATS_SIM = {
    key: (int if key in DATA_LABLES_SIM[:10] else float) for key in DATA_LABLES_SIM
}

IDX_STATES_DATA_SPARSE = np.array(
    [
        0,  # Agent_ID
        1,  # Periods
        2,  # Education Level
        3,  # Lagged Choice
        4,  # Exp Part time
        5,  # Exp Full time
        7,  # Age_Youngest_Child
    ],
    dtype=int,
)

LABELS_DATA_SPARSE = [
    label for i, label in enumerate(DATA_LABLES_SIM) if i in IDX_STATES_DATA_SPARSE
] + ["Choice", "Wage_Observed"]

DATA_FORMATS_SPARSE_1 = {
    "Identifier": int,
    "Wage_Observed": float,
}

DATA_FORMATS_SPARSE_2 = {key: np.int8 for key in LABELS_DATA_SPARSE[1:-1]}
DATA_FORMATS_SPARSE = {**DATA_FORMATS_SPARSE_1, **DATA_FORMATS_SPARSE_2}
