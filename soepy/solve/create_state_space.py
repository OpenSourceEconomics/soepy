import numba
import numpy as np

from soepy.exogenous_processes.children import define_child_age_update_rule
from soepy.shared.shared_constants import MISSING_INT
from soepy.shared.shared_constants import NUM_CHOICES
from soepy.solve.covariates import construct_covariates


def create_state_space_objects(model_spec):
    states, indexer = pyth_create_state_space(model_spec)
    covariates = construct_covariates(states, model_spec)

    child_age_update_rule = define_child_age_update_rule(model_spec, states)

    child_state_indexes = create_child_indexes(
        states, indexer, model_spec, child_age_update_rule
    )
    return states, indexer, covariates, child_age_update_rule, child_state_indexes


def _kid_age_to_index(age_kid: int, kids_ages_len: int) -> int:
    """
    Original code indexes indexer[..., age_kid, ...] where age_kid can be -1.
    In NumPy, -1 means "last element". That is what the original does.
    We preserve that convention:
      - age_kid == -1 -> index kids_ages_len - 1
      - else -> index age_kid
    """
    return kids_ages_len - 1 if age_kid == -1 else age_kid


def pyth_create_state_space(model_spec):
    """
    Pure NumPy version of pyth_create_state_space (no numba).
    Returns:
        states  : (N,8) int array
        indexer : ndarray with same shape as original
    """
    kids_ages = np.arange(-1, model_spec.child_age_max + 1)
    A = kids_ages.shape[0]

    max_exp = (
        model_spec.num_periods + model_spec.init_exp_max
    )  # inclusive max exp value allowed by indexer dims

    shape = (
        model_spec.num_periods,
        model_spec.num_educ_levels,
        NUM_CHOICES,
        max_exp,
        max_exp,
        model_spec.num_types,
        A,
        2,
    )
    indexer = np.full(shape, MISSING_INT, dtype=np.int32)

    # Precompute full experience grid once; we’ll filter it per (period, educ)
    exp_vals = np.arange(max_exp, dtype=np.int32)
    EP, EF = np.meshgrid(
        exp_vals, exp_vals, indexing="ij"
    )  # EP: (max_exp,max_exp), EF: same

    blocks = []
    i = 0

    for period in range(model_spec.num_periods):
        for type_ in range(model_spec.num_types):
            for partner in range(2):
                for age_kid in kids_ages:
                    # replicate original kid timing restrictions
                    if age_kid - model_spec.child_age_init_max > period:
                        continue

                    if (
                        period > model_spec.last_child_bearing_period
                        and 0
                        <= age_kid
                        <= min(period - (model_spec.last_child_bearing_period + 1), 10)
                    ):
                        continue

                    age_idx = _kid_age_to_index(int(age_kid), A)

                    for educ_level in range(model_spec.num_educ_levels):
                        edu_years = int(model_spec.educ_years[educ_level])

                        # has she completed education already?
                        if edu_years > period:
                            continue

                        # Basic feasibility region for experiences (vectorized):
                        # exp_f + exp_p <= period + 2*init_exp_max - edu_years
                        max_total = period + 2 * model_spec.init_exp_max - edu_years
                        # also must be <= period + init_exp_max individually (this is already implied by max_exp axis),
                        # but original additionally checks exp_f > period + init_exp_max etc.
                        max_ind = period + model_spec.init_exp_max

                        feasible = (EP + EF) <= max_total
                        feasible &= (EF <= max_ind) & (EP <= max_ind)

                        # Extract feasible pairs as 1D arrays
                        fp = EP[feasible].astype(np.int32, copy=False)
                        ff = EF[feasible].astype(np.int32, copy=False)

                        if fp.size == 0:
                            continue

                        if period == edu_years:
                            # Entry-period: original code adds states for ALL lagged choices, no extra restrictions.
                            lagged = np.tile(
                                np.arange(NUM_CHOICES, dtype=np.int32), fp.size
                            )
                            exp_p_rep = np.repeat(fp, NUM_CHOICES)
                            exp_f_rep = np.repeat(ff, NUM_CHOICES)

                        else:
                            # Non-entry periods: apply the lagged-choice restrictions (vectorized).
                            max_ft = period + model_spec.init_exp_max - edu_years
                            max_pt = period + model_spec.init_exp_max - edu_years

                            # allowed[c, j] means for pair c (fp[c], ff[c]) lagged choice j is allowed
                            allowed = np.ones((fp.size, NUM_CHOICES), dtype=bool)

                            # only worked full-time -> lagged must be 2
                            mask_only_ft = ff == max_ft
                            allowed[mask_only_ft, :] = False
                            allowed[mask_only_ft, 2] = True

                            # only worked part-time -> lagged must be 1
                            mask_only_pt = fp == max_pt
                            allowed[mask_only_pt, :] = False
                            allowed[mask_only_pt, 1] = True

                            # never worked full-time -> cannot have lagged 2
                            allowed[(ff == 0), 2] = False
                            # never worked part-time -> cannot have lagged 1
                            allowed[(fp == 0), 1] = False

                            # always employed -> cannot have lagged 0
                            allowed[(fp + ff == max_total), 0] = False

                            # Build rows by expanding only allowed (pair, lagged) combinations
                            pair_idx, lagged = np.nonzero(allowed)
                            if lagged.size == 0:
                                continue
                            exp_p_rep = fp[pair_idx]
                            exp_f_rep = ff[pair_idx]
                            lagged = lagged.astype(np.int32, copy=False)

                        n = lagged.size

                        # Create block of states (N,8)
                        block = np.empty((n, 8), dtype=np.int32)
                        block[:, 0] = period
                        block[:, 1] = educ_level
                        block[:, 2] = lagged
                        block[:, 3] = exp_p_rep
                        block[:, 4] = exp_f_rep
                        block[:, 5] = type_
                        block[:, 6] = age_kid  # keep original "value", including -1
                        block[:, 7] = partner

                        blocks.append(block)

                        # Fill indexer with consecutive ids [i, i+n)
                        ids = np.arange(i, i + n, dtype=np.int32)

                        # IMPORTANT: age axis index must follow the original -1 -> last convention
                        age_index_for_indexer = age_idx  # already mapped

                        indexer[
                            period,
                            educ_level,
                            lagged,
                            exp_p_rep,
                            exp_f_rep,
                            type_,
                            age_index_for_indexer,
                            partner,
                        ] = ids

                        i += n

    states = (
        np.concatenate(blocks, axis=0) if blocks else np.empty((0, 8), dtype=np.int32)
    )
    return states, indexer


def create_child_indexes(states, indexer, model_spec, child_age_update_rule):
    """
    Vectorized replacement for create_child_indexes + get_child_states_index.
    Output shape: (num_states, NUM_CHOICES, 2, 2) filled with MISSING_INT where invalid.
    """
    child_indexes = np.full(
        (states.shape[0], NUM_CHOICES, 2, 2), MISSING_INT, dtype=np.int32
    )
    if states.shape[0] == 0:
        return child_indexes

    period = states[:, 0].astype(np.int32, copy=False)
    educ = states[:, 1].astype(np.int32, copy=False)
    lag0 = states[:, 2].astype(np.int32, copy=False)
    exp_p = states[:, 3].astype(np.int32, copy=False)
    exp_f = states[:, 4].astype(np.int32, copy=False)
    type_ = states[:, 5].astype(np.int32, copy=False)
    age_kid_val = states[:, 6].astype(np.int32, copy=False)
    partner = states[:, 7].astype(np.int32, copy=False)

    A = indexer.shape[6]  # kid-age axis length
    age_idx = np.where(age_kid_val == -1, A - 1, age_kid_val).astype(
        np.int32, copy=False
    )

    valid = period < (model_spec.num_periods - 1)
    if not np.any(valid):
        return child_indexes

    v = np.where(valid)[0]
    next_period = (period[v] + 1).astype(np.int32, copy=False)

    # Find parent index k_parent and implied child age update rule for "no new child" branch
    k_parent = indexer[
        period[v],
        educ[v],
        lag0[v],
        exp_p[v],
        exp_f[v],
        type_[v],
        age_idx[v],
        partner[v],
    ]
    rule = child_age_update_rule[k_parent].astype(np.int32, copy=False)

    # Helper to fill for action a:
    # - action 0: exp doesn't change
    # - action 1: exp_p + 1
    # - action 2: exp_f + 1
    for a in range(NUM_CHOICES):
        ep = exp_p[v] + (1 if a == 1 else 0)
        ef = exp_f[v] + (1 if a == 2 else 0)

        # branch 0: use updated child age (rule), partner shock in {0,1}
        child_indexes[v, a, 0, 0] = indexer[
            next_period, educ[v], a, ep, ef, type_[v], rule, 0
        ]
        child_indexes[v, a, 0, 1] = indexer[
            next_period, educ[v], a, ep, ef, type_[v], rule, 1
        ]

        # branch 1: child arrives -> kid age resets to 0, partner shock in {0,1}
        child_indexes[v, a, 1, 0] = indexer[
            next_period, educ[v], a, ep, ef, type_[v], 0, 0
        ]
        child_indexes[v, a, 1, 1] = indexer[
            next_period, educ[v], a, ep, ef, type_[v], 0, 1
        ]

    return child_indexes
