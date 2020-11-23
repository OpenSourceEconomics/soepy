"""This module reads in information on probabilities regarding the initial conditions
on years of education"""
import pandas as pd


def gen_prob_educ_level_vector(model_spec):
    """Generates a vector of probabilities reflecting the fractions of individuals
    with low, middle, and high levels of education in the SOEP data."""

    prob_educ_level = pd.read_pickle(model_spec.educ_shares_file_name)

    prob_educ_level = list(prob_educ_level["educ_shares"])

    return prob_educ_level
