"""Initial experience distributions.

The continuous-experience refactor uses a single experience stock. For simulation and
initial conditions we still need distributions over *experience years*.

We keep the legacy inputs (separate PT and FT experience share files) but expose them
explicitly:

- ``gen_prob_init_exp_component_vector`` reads one share file and returns a
  distribution over years 0..init_exp_max by education.
- ``gen_prob_init_exp_years_vector`` combines PT and FT distributions via
  convolution to obtain a distribution over total experience years
  0..(2*init_exp_max) by education.

No implicit defaults: the model spec must define
``ft_exp_shares_file_name`` and ``pt_exp_shares_file_name``.
"""
import numpy as np
import pandas as pd


def gen_prob_init_exp_component_vector(model_spec, model_spec_exp_file_key):
    """Generates a list of lists containing the shares of individuals with
    ft/pt experience of 0, 1, 2, 3, and 4 years in the model's first period.
    Shares differ by the level of education of the individuals."""

    exp_shares = pd.read_pickle(model_spec_exp_file_key)

    init_exp = []
    for educ_level in range(model_spec.num_educ_levels):
        exp_shares_list = exp_shares[
            exp_shares.index.get_level_values("educ_level") == educ_level
        ]["exper_shares"].to_list()
        exp_shares_list[0] = 1 - sum(exp_shares_list[1:])
        init_exp.append(exp_shares_list)

    return init_exp


def gen_prob_init_exp_years_vector(model_spec):
    """Generate distribution over total initial experience years by education."""

    prob_ft = gen_prob_init_exp_component_vector(
        model_spec, model_spec.ft_exp_shares_file_name
    )
    prob_pt = gen_prob_init_exp_component_vector(
        model_spec, model_spec.pt_exp_shares_file_name
    )

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
