"""This module contains constants, indices etc. which are constant throughout the project.

State vector layout (columns):

0. period
1. educ_level
2. lagged_choice
3. type
4. age_youngest_child
5. partner
"""
import numpy as np

# Set values of constants used across all modules here
MISSING_INT = -99
INVALID_FLOAT = -99.0
NUM_CHOICES = 3
# Hours worked per month
# Assumption: weekly working hours times 4.5 weeks in a month
HOURS = np.array([0, 18, 38])
PERIOD = 0
EDUC_LEVEL = 1
LAGGED_CHOICE = 2
TYPE = 3
AGE_YOUNGEST_CHILD = 4
PARTNER = 5
N_STATE_VARS = 6
