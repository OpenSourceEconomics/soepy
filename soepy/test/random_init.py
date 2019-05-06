"""This function provides an random init file generating process."""
import collections

import numpy as np
import yaml

from soepy.python.pre_processing.model_processing import expand_init_dict


def random_init(constr=None):
    """The module provides a random dictionary generating process for test purposes."""

    # Check for pre specified constraints
    if constr is not None:
        pass
    else:
        constr = {}

    if "EDUC_MAX" in constr.keys():
        educ_max = constr["EDUC_MAX"]
    else:
        educ_max = 12

    if "EDUC_MIN" in constr.keys():
        educ_min = constr["EDUC_MIN"]
    else:
        educ_min = 10

    if "AGENTS" in constr.keys():
        agents = constr["AGENTS"]
    else:
        agents = np.random.randint(500, 1000)

    if "PERIODS" in constr.keys():
        periods = constr["PERIODS"]
    else:
        periods = np.random.randint(3, 6)

    if "SEED_SIM" in constr.keys():
        seed_sim = constr["SEED_SIM"]
    else:
        seed_sim = np.random.randint(1000, 9999)

    if "SEED_EMAX" in constr.keys():
        seed_emax = constr["SEED_EMAX"]
    else:
        seed_emax = np.random.randint(1000, 9999)

    if "NUM_DRAWS_EMAX" in constr.keys():
        num_draws_emax = constr["NUM_DRAWS_EMAX"]
    else:
        num_draws_emax = np.random.randint(400, 600)

    init_dict = dict()

    for key_ in [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "PARAMETERS",
    ]:
        init_dict[key_] = {}

    init_dict["GENERAL"]["num_periods"] = periods

    init_dict["CONSTANTS"]["delta"] = np.random.uniform(0.8, 0.99)
    init_dict["CONSTANTS"]["mu"] = np.random.uniform(-0.7, -0.4)
    init_dict["CONSTANTS"]["benefits"] = np.random.uniform(1600.0, 2200.0)

    init_dict["INITIAL_CONDITIONS"]["educ_max"] = educ_max
    init_dict["INITIAL_CONDITIONS"]["educ_min"] = educ_min

    init_dict["SIMULATION"]["seed_sim"] = seed_sim
    init_dict["SIMULATION"]["num_agents_sim"] = agents

    init_dict["SOLUTION"]["seed_emax"] = seed_emax
    init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax

    init_dict["PARAMETERS"]["gamma_0s1"], init_dict["PARAMETERS"][
        "gamma_0s2"
    ], init_dict["PARAMETERS"]["gamma_0s3"] = np.random.uniform(6.0, 1.0, 3).tolist()

    init_dict["PARAMETERS"]["gamma_1s1"], init_dict["PARAMETERS"][
        "gamma_1s2"
    ], init_dict["PARAMETERS"]["gamma_1s3"] = np.random.uniform(0.2, 0.3, 3).tolist()

    init_dict["PARAMETERS"]["g_s1"], init_dict["PARAMETERS"]["g_s2"], init_dict[
        "PARAMETERS"
    ]["g_s3"] = np.random.uniform(0.02, 0.5, 3).tolist()

    init_dict["PARAMETERS"]["delta_s1"], init_dict["PARAMETERS"]["delta_s2"], init_dict[
        "PARAMETERS"
    ]["delta_s3"] = np.random.uniform(0.1, 0.9, 3).tolist()

    init_dict["PARAMETERS"]["theta_p"], init_dict["PARAMETERS"][
        "theta_f"
    ] = np.random.uniform(-0.5, -0.1, 2).tolist()

    init_dict["PARAMETERS"]["sigma_1"], init_dict["PARAMETERS"]["sigma_2"], init_dict[
        "PARAMETERS"
    ]["sigma_3"] = np.random.uniform(1.0, 2.0, 3).tolist()

    print_dict(init_dict)

    return init_dict


def print_dict(init_dict, file_name="test"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "PARAMETERS",
    ]
    for key_ in order:
        ordered_dict[key_] = init_dict[key_]

    with open("{}.soepy.yml".format(file_name), "w") as outfile:
        yaml.dump(ordered_dict, outfile, explicit_start=True, indent=4)


def namedtuple_to_dict(named_tuple):
    """Converts named tuple to flat dictionary"""

    init_dict_flat = dict(named_tuple._asdict())

    return init_dict_flat


def init_dict_flat_to_init_dict(init_dict_flat):
    """Converts flattened init dict to init dict structure
    as in the init file"""

    init_dict = dict()

    init_dict["GENERAL"] = dict()
    init_dict["GENERAL"]["num_periods"] = init_dict_flat["num_periods"]

    init_dict["CONSTANTS"] = dict()
    init_dict["CONSTANTS"]["delta"] = init_dict_flat["delta"]
    init_dict["CONSTANTS"]["mu"] = init_dict_flat["mu"]
    init_dict["CONSTANTS"]["benefits"] = init_dict_flat["benefits"]

    init_dict["INITIAL_CONDITIONS"] = dict()
    init_dict["INITIAL_CONDITIONS"]["educ_max"] = init_dict_flat["educ_max"]
    init_dict["INITIAL_CONDITIONS"]["educ_min"] = init_dict_flat["educ_min"]

    init_dict["SIMULATION"] = dict()
    init_dict["SIMULATION"]["seed_sim"] = init_dict_flat["seed_sim"]
    init_dict["SIMULATION"]["num_agents_sim"] = init_dict_flat["num_agents_sim"]

    init_dict["SOLUTION"] = dict()
    init_dict["SOLUTION"]["seed_emax"] = init_dict_flat["seed_emax"]
    init_dict["SOLUTION"]["num_draws_emax"] = init_dict_flat["num_draws_emax"]

    init_dict["PARAMETERS"] = dict()
    init_dict["PARAMETERS"]["gamma_0s1"], init_dict["PARAMETERS"][
        "gamma_0s2"
    ], init_dict["PARAMETERS"]["gamma_0s3"] = init_dict_flat["gamma_0s"]
    init_dict["PARAMETERS"]["gamma_1s1"], init_dict["PARAMETERS"][
        "gamma_1s2"
    ], init_dict["PARAMETERS"]["gamma_1s3"] = init_dict_flat["gamma_1s"]
    init_dict["PARAMETERS"]["g_s1"], init_dict["PARAMETERS"]["g_s2"], init_dict[
        "PARAMETERS"
    ]["g_s3"] = init_dict_flat["g_s"]
    init_dict["PARAMETERS"]["delta_s1"], init_dict["PARAMETERS"]["delta_s2"], init_dict[
        "PARAMETERS"
    ]["delta_s3"] = init_dict_flat["delta_s"]
    init_dict["PARAMETERS"]["theta_p"] = init_dict_flat["theta_p"]
    init_dict["PARAMETERS"]["theta_f"] = init_dict_flat["theta_f"]
    init_dict["PARAMETERS"]["sigma_1"], init_dict["PARAMETERS"]["sigma_2"], init_dict[
        "PARAMETERS"
    ]["sigma_3"] = init_dict_flat["sigma"]

    init_dict["DERIVED_ATTR"] = dict()
    init_dict["DERIVED_ATTR"]["educ_range"] = init_dict_flat["educ_range"]
    init_dict["DERIVED_ATTR"]["shocks_cov"] = init_dict_flat["shocks_cov"]

    return init_dict


def read_init_file2(init_file_name):
    """Loads the model specification from yaml file
    as dictionary and expands it without further changes"""

    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict_base = yaml.load(y, Loader=yaml.FullLoader)

    init_dict_expanded = expand_init_dict(init_dict_base)

    return init_dict_expanded
