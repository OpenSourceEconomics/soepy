import pickle
from soepy.soepy_config import TEST_RESOURCES_DIR


def gen_prob_educ_years_vector(model_spec):
    """ Generates a vector of probabilities reflecting the fractions of individuals
    with low, middle, and high levels of education in the SOEP data."""

    if "observed" in str(model_spec.educ_info_file_name).split("."):
        with open(str(model_spec.educ_info_file_name), "rb") as f:
            prob_educ_years = pickle.load(f)

    else:
        with open(
            str(TEST_RESOURCES_DIR) + "/" + str(model_spec.educ_info_file_name), "rb"
        ) as f:
            prob_educ_years = pickle.load(f)

    return prob_educ_years
