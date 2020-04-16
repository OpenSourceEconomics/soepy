import pandas as pd


def gen_prob_educ_years_vector(model_spec):
    """ Generates a vector of probabilities reflecting the fractions of individuals
    with low, middle, and high levels of education in the SOEP data."""

    prob_educ_years_df = pd.read_pickle(str(model_spec.educ_info_file_name))

    prob_educ_years = list(prob_educ_years_df["Fraction"])

    return prob_educ_years
