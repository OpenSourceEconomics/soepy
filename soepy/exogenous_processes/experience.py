"""This module reads in information on probabilities to have accumulated experience
in part-time and/or full-time work before the model entry/initial age."""
import pandas as pd

def gen_prob_init_exp_vector(model_spec):
    """Generates a list of lists containing the shares of individuals with
    ft/pt experience of 0, 1, 2, 3, and 4 years in the model's first period.
    Shares differ by the level of education of the individuals."""

    exp_shares = pd.read_pickle(model_spec.ft_exp_shares_file_name)

    init_exp = []
    for educ_level in range(model_spec.num_educ_levels):
        exp_shares_list = exp_shares[
            exp_shares.index.get_level_values("educ_level") == educ_level
        ]["exper_shares"].to_list()
        exp_shares_list[0] = 1 - sum(exp_shares_list[1:])
        init_exp.append(exp_shares_list)

    return init_exp
