import numpy as np


def lagged_choice_initial(initial_exp_years):
    """Determine initial lagged choice from total experience.

    Rule (per project decision):
    - lagged_choice = 2 (full-time) if initial experience years > 1
    - otherwise lagged_choice = 0
    """

    lagged_choice = np.zeros_like(initial_exp_years, dtype=int)
    lagged_choice[initial_exp_years > 1] = 2
    return lagged_choice
