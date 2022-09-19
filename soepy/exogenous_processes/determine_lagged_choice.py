import numpy as np


def lagged_choice_initial(ft_exp, pt_exp):
    past_ft = ft_exp > 0
    past_pt = pt_exp > 0
    only_ft = past_pt & ~past_pt
    only_pt = ~past_ft & past_pt
    both = past_ft & past_pt
    probs_ft = ft_exp[both] / (ft_exp[both] + pt_exp[both])
    ft_drawn_sub = ((np.random.uniform(size=probs_ft.shape[0]) <= probs_ft) * 1) == 1

    # Create boolean array
    ft_drawn = np.zeros_like(both)
    ft_drawn[both] = ft_drawn_sub

    # Create output array
    lagged_choice = np.zeros_like(ft_exp)

    # Assign
    lagged_choice[only_pt] = 1
    lagged_choice[both & ~ft_drawn] = 1

    lagged_choice[only_ft] = 2
    lagged_choice[both & ft_drawn] = 2

    return lagged_choice
