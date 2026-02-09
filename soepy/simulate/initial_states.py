import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import NUM_CHOICES


def validate_initial_states(initial_states, model_spec):
    required_columns = [
        "Identifier",
        "Period",
        "Education_Level",
        "Lagged_Choice",
        "Experience_Part_Time",
        "Experience_Full_Time",
        "Type",
        "Age_Youngest_Child",
        "Partner_Indicator",
    ]
    missing = [col for col in required_columns if col not in initial_states.columns]
    if missing:
        raise ValueError(f"Initial states missing columns: {missing}")

    if initial_states[required_columns].isna().any().any():
        raise ValueError("Initial states contain missing values.")

    identifiers = initial_states["Identifier"].to_numpy()
    unique_ids = np.unique(identifiers)
    if len(unique_ids) != len(identifiers):
        raise ValueError("Initial state identifiers must be unique.")

    if not np.issubdtype(unique_ids.dtype, np.integer):
        raise ValueError("Initial state identifiers must be integers.")

    expected_ids = np.arange(len(unique_ids), dtype=unique_ids.dtype)
    if not np.array_equal(np.sort(unique_ids), expected_ids):
        raise ValueError("Initial state identifiers must be consecutive from 0.")

    # if (initial_states["Period"] < 0).any() or (
    #     initial_states["Period"] >= model_spec.num_periods
    # ).any():
    #     raise ValueError("Initial state periods out of bounds.")

    if (initial_states["Education_Level"] < 0).any() or (
        initial_states["Education_Level"] >= model_spec.num_educ_levels
    ).any():
        raise ValueError("Initial state education levels out of bounds.")

    if (initial_states["Type"] < 0).any() or (
        initial_states["Type"] >= model_spec.num_types
    ).any():
        raise ValueError("Initial state types out of bounds.")

    if (initial_states["Lagged_Choice"] < 0).any() or (
        initial_states["Lagged_Choice"] >= NUM_CHOICES
    ).any():
        raise ValueError("Initial state lagged choices out of bounds.")

    return initial_states
