"""Column indices for the state space array.

The continuous-experience refactor uses a discrete state space without part-time or
full-time experience dimensions.

State vector layout (columns):

0. period
1. educ_level
2. lagged_choice
3. type
4. age_youngest_child
5. partner
"""

PERIOD = 0
EDUC_LEVEL = 1
LAGGED_CHOICE = 2
TYPE = 3
AGE_YOUNGEST_CHILD = 4
PARTNER = 5

N_STATE_VARS = 6
