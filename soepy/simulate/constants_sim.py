import numpy as np

DATA_LABLES_SIM_int = [
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
]
DATA_LABLES_SIM_float = [
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
DATA_LABLES_SIM = [*DATA_LABLES_SIM_int, *DATA_LABLES_SIM_float]

STATE_LABELS_SIM = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Stock",
    "Type",
    "Age_Youngest_Child",
    "Partner_Indicator",
]

DATA_FORMATS_SIM = {
    **{key: np.int8 for key in DATA_LABLES_SIM_int},
    **{key: float for key in DATA_LABLES_SIM_float},
}

# Minimal dataset used in estimation/regressions.
LABELS_SPARSE_INT8 = [
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Experience_Stock",
    "Age_Youngest_Child",
    "Choice",
]

DATA_FORMATS_OTHER = {
    "Identifier": int,
    "Wage_Observed": float,
    "Experience_Stock": float,
}
LABELS_DATA_SPARSE = [*LABELS_SPARSE_INT8, *DATA_FORMATS_OTHER.keys()]

DATA_FORMATS_INT8 = {key: np.int8 for key in LABELS_SPARSE_INT8}
DATA_FORMATS_SPARSE = {**DATA_FORMATS_INT8, **DATA_FORMATS_OTHER}
