"""Initial experience distribution.

The continuous-experience refactor uses a single experience stock. We still draw an
initial experience *in years* and then convert it to a stock using
``exp_years_to_stock``.

For backwards compatibility with existing input data, we allow providing separate PT
and FT experience share files and combine them as

    exp_years_init = exp_ft + exp_pt

This discards the composition (PT vs FT) which is no longer part of the state.
"""
import numpy as np
import pandas as pd


def _read_init_exp_shares_by_educ(model_spec, exp_file_name):
    exp_shares = pd.read_pickle(exp_file_name)

    init_exp = []
    for educ_level in range(model_spec.num_educ_levels):
        shares = exp_shares[
            exp_shares.index.get_level_values("educ_level") == educ_level
        ]["exper_shares"].tolist()
        shares[0] = 1 - sum(shares[1:])
        init_exp.append(shares)

    return init_exp


def gen_prob_init_exp_years_vector(model_spec):
    """Generate a distribution over initial experience years by education.

    If both ``ft_exp_shares_file_name`` and ``pt_exp_shares_file_name`` exist on the
    model spec, we combine them by convolution.

    Returns
    -------
    list[list[float]]
        Outer list over education, inner list over years 0..(2*init_exp_max).
    """

    ft_file = getattr(model_spec, "ft_exp_shares_file_name", None)
    pt_file = getattr(model_spec, "pt_exp_shares_file_name", None)

    if ft_file is None or pt_file is None:
        raise AttributeError(
            "model_spec must define 'ft_exp_shares_file_name' and 'pt_exp_shares_file_name'"
        )

    prob_ft = _read_init_exp_shares_by_educ(model_spec, ft_file)
    prob_pt = _read_init_exp_shares_by_educ(model_spec, pt_file)

    max_years = 2 * model_spec.init_exp_max
    out = []

    for educ_level in range(model_spec.num_educ_levels):
        p_ft = np.asarray(prob_ft[educ_level], dtype=float)
        p_pt = np.asarray(prob_pt[educ_level], dtype=float)

        p = np.convolve(p_ft, p_pt)
        p = p[: max_years + 1]
        p = p / p.sum()
        out.append(p.tolist())

    return out
