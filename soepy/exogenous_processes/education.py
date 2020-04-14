import pandas as pd

from soepy.soepy_config import TEST_RESOURCES_DIR


def gen_prob_educ_years_vector(model_spec):
    """ Generates a vector of probabilities reflecting the fractions of individuals
    with low, middle, and high levels of education in the SOEP data."""

    if "observed" in str(model_spec.educ_info_file_name).split("."):
        prob_educ_years_df = pd.read_pickle(str(model_spec.educ_info_file_name))
    else:
        prob_educ_years_df = pd.read_pickle(
            str(TEST_RESOURCES_DIR) + "/" + str(model_spec.educ_info_file_name)
        )

    prob_educ_years = list(prob_educ_years_df["Fraction"])

    return prob_educ_years
