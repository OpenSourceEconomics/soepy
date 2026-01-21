import numpy as np

DATA_LABLES_SIM = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Experience_Stock",
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
    key: (int if key in DATA_LABLES_SIM[:11] else float) for key in DATA_LABLES_SIM
}

# Minimal dataset used in estimation/regressions.
LABELS_STATE_SPARSE = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Experience_Stock",
    "Age_Youngest_Child",
]

LABELS_DATA_SPARSE = LABELS_STATE_SPARSE + ["Choice", "Wage_Observed"]

DATA_FORMATS_SPARSE_1 = {
    "Identifier": int,
    "Wage_Observed": float,
}

DATA_FORMATS_SPARSE_2 = {key: np.int8 for key in LABELS_DATA_SPARSE[1:-1]}
DATA_FORMATS_SPARSE = {**DATA_FORMATS_SPARSE_1, **DATA_FORMATS_SPARSE_2}
