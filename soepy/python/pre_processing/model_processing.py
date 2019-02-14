import yaml
import numpy as np


def read_init_file(init_file_name):
    """Reads in the model specification from yaml file."""
    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict = yaml.load(y)

    attr_dict = init_dict_to_attr_dict(init_dict)

    return attr_dict


def init_dict_to_attr_dict(init_dict):
    """Enhances read in initialization dictionary to
    an attribute dictionary to be used by all further model funtions."""

    # Calculate range of years of education in the (simulated) sample
    educ_min = init_dict["INITIAL_CONDITIONS"]["educ_min"]
    educ_max = init_dict["INITIAL_CONDITIONS"]["educ_max"]
    educ_range = educ_max - educ_min + 1

    # Calculate covariances of the error terms given standard diviations
    shocks_cov = init_dict["PARAMETERS"]["optim_paras"][14:17]
    shocks_cov = [shocks_cov[0] ** 2, shocks_cov[1] ** 2, shocks_cov[2] ** 2]

    init_dict["DERIVED_ATTR"] = {"educ_range": educ_range, "shocks_cov": shocks_cov}

    # Return function output
    return init_dict
