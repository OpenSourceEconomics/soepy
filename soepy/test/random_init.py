"""This function provides an random init file generating process."""
import collections

import numpy as np
import oyaml as yaml

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
        educ_max = 14

    if "EDUC_MIN" in constr.keys():
        educ_max = constr["EDUC_MIN"]
    else:
        educ_min = 10

    if "AGENTS" in constr.keys():
        agents = constr["AGENTS"]
    else:
        agents = np.random.randint(500, 1000)

    if "PERIODS" in constr.keys():
        periods = constr["PERIODS"]
    else:
        periods = np.random.randint(1, 5)

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
    init_dict["GENERAL"]["num_choices"] = 3

    init_dict["CONSTANTS"]["delta"] = np.random.uniform(0.8, 0.99)
    init_dict["CONSTANTS"]["mu"] = np.random.uniform(-0.7, -0.4)
    init_dict["CONSTANTS"]["benefits"] = np.random.uniform(1600.0, 2200.0)

    init_dict["INITIAL_CONDITIONS"]["educ_max"] = educ_max
    init_dict["INITIAL_CONDITIONS"]["educ_min"] = educ_min

    init_dict["SIMULATION"]["seed_sim"] = seed_sim
    init_dict["SIMULATION"]["num_agents_sim"] = agents

    init_dict["SOLUTION"]["seed_emax"] = seed_emax
    init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax

    # Generate random parameterization
    gamma0 = np.random.normal(6.0, 1.0, 3)
    gamma1 = np.random.uniform(0.2, 0.3, 3)
    g_s = np.random.uniform(0.02, 0.5, 3)
    delta = np.random.uniform(0.1, 0.9, 3)
    theta = np.random.uniform(-0.5, -0.1, 2)
    sigma = np.random.uniform(1.0, 2.0, 3)

    init_dict["PARAMETERS"]["optim_paras"] = np.concatenate(
        (gamma0, gamma1, g_s, delta, theta, sigma)
    ).tolist()

    init_dict["PARAMETERS"]["order"] = [
        "gamma_0s1",
        "gamma_0s2",
        "gamma_0s3",
        "gamma_1s1",
        "gamma_1s2",
        "gamma_1s3",
        "g_s1",
        "g_s2",
        "g_s3",
        "delta_s1",
        "delta_s2",
        "delta_s3",
        "theta_p",
        "theta_f",
        "sigma_0",
        "sigma_1",
        "sigma_2",
    ]

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


def namedtuple_to_dict(model_params):

    """Transfers model specification from a
    named tuple class object to dictionary."""

    init_dict = {}

    init_dict["GENERAL"] = {}
    init_dict["GENERAL"]["num_periods"] = model_params.num_periods
    init_dict["GENERAL"]["num_choices"] = model_params.num_choices

    init_dict["CONSTANTS"] = {}
    init_dict["CONSTANTS"]["delta"] = model_params.delta
    init_dict["CONSTANTS"]["mu"] = model_params.mu
    init_dict["CONSTANTS"]["benefits"] = model_params.benefits

    init_dict["INITIAL_CONDITIONS"] = {}
    init_dict["INITIAL_CONDITIONS"]["educ_max"] = model_params.educ_max
    init_dict["INITIAL_CONDITIONS"]["educ_min"] = model_params.educ_min

    init_dict["SIMULATION"] = {}
    init_dict["SIMULATION"]["seed_sim"] = model_params.seed_sim
    init_dict["SIMULATION"]["num_agents_sim"] = model_params.num_agents_sim

    init_dict["SOLUTION"] = {}
    init_dict["SOLUTION"]["seed_emax"] = model_params.seed_emax
    init_dict["SOLUTION"]["num_draws_emax"] = model_params.num_draws_emax

    init_dict["PARAMETERS"] = {}
    init_dict["PARAMETERS"]["optim_paras"] = model_params.optim_paras

    init_dict["DERIVED_ATTR"] = {}
    init_dict["DERIVED_ATTR"]["educ_range"] = model_params.educ_range
    init_dict["DERIVED_ATTR"]["shocks_cov"] = model_params.shocks_cov

    return init_dict


def read_init_file2(init_file_name):
    """Reads in the model specification from yaml file."""

    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    init_dict = expand_init_dict(init_dict)

    return init_dict
