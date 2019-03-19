import collections as cl

import oyaml as yaml


def read_init_file(init_file_name):
    """Reads in the model specification from yaml file."""

    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    init_dict = expand_init_dict(init_dict)

    model_params = create_namedtuple(init_dict)

    return model_params


def expand_init_dict(init_dict):
    """Enhances read in initialization dictionary by
    adding model parameters derived from the
    specified initialisation file"""

    # Calculate range of years of education in the (simulated) sample
    educ_min = init_dict["INITIAL_CONDITIONS"]["educ_min"]
    educ_max = init_dict["INITIAL_CONDITIONS"]["educ_max"]
    educ_range = educ_max - educ_min + 1

    # Calculate covariances of the error terms given standard deviations
    shocks_cov = init_dict["PARAMETERS"]["optim_paras"][14:17]
    shocks_cov = [shocks_cov[0] ** 2, shocks_cov[1] ** 2, shocks_cov[2] ** 2]

    init_dict["DERIVED_ATTR"] = {"educ_range": educ_range, "shocks_cov": shocks_cov}

    # Return function output
    return init_dict


def create_namedtuple(init_dict):
    """Transfers model specification from a dictionary
    to a named tuple class object."""

    model_params = cl.namedtuple("model_parameters", "")
    model_params.num_periods = init_dict["GENERAL"]["num_periods"]
    model_params.num_choices = init_dict["GENERAL"]["num_choices"]

    model_params.delta = init_dict["CONSTANTS"]["delta"]
    model_params.mu = init_dict["CONSTANTS"]["mu"]
    model_params.benefits = init_dict["CONSTANTS"]["benefits"]

    model_params.educ_max = init_dict["INITIAL_CONDITIONS"]["educ_max"]
    model_params.educ_min = init_dict["INITIAL_CONDITIONS"]["educ_min"]

    model_params.seed_sim = init_dict["SIMULATION"]["seed_sim"]
    model_params.num_agents_sim = init_dict["SIMULATION"]["num_agents_sim"]

    model_params.seed_emax = init_dict["SOLUTION"]["seed_emax"]
    model_params.num_draws_emax = init_dict["SOLUTION"]["num_draws_emax"]

    model_params.optim_paras = init_dict["PARAMETERS"]["optim_paras"]

    model_params.educ_range = init_dict["DERIVED_ATTR"]["educ_range"]
    model_params.shocks_cov = init_dict["DERIVED_ATTR"]["shocks_cov"]

    return model_params
