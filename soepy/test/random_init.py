"""This function provides an random init file generating process."""
import collections

import numpy as np
import pandas as pd
import yaml


def random_init(constr=None):
    """The module provides a random dictionary generating process for test purposes.
    It generates a random model_spec init dictionary and a random model_params init
    dataframe."""

    # Generate model spec init dict

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

    if "LOW_BOUND" in constr.keys():
        low_bound = constr["LOW_BOUND"]
    else:
        low_bound = 10

    if "MIDDLE_BOUND" in constr.keys():
        middle_bound = constr["MIDDLE_BOUND"]
    else:
        middle_bound = 11

    if "HIGH_BOUND" in constr.keys():
        high_bound = constr["HIGH_BOUND"]
    else:
        high_bound = 12

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

    model_spec_init_dict = dict()

    for key_ in [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "EDUC_LEVEL_BOUNDS",
        "SIMULATION",
        "SOLUTION",
    ]:
        model_spec_init_dict[key_] = {}

    model_spec_init_dict["GENERAL"]["num_periods"] = periods

    model_spec_init_dict["CONSTANTS"]["delta"] = np.random.uniform(0.8, 0.99)
    model_spec_init_dict["CONSTANTS"]["mu"] = np.random.uniform(-0.7, -0.4)
    model_spec_init_dict["CONSTANTS"]["benefits"] = np.random.uniform(4.0, 7.0)

    model_spec_init_dict["INITIAL_CONDITIONS"]["educ_max"] = educ_max
    model_spec_init_dict["INITIAL_CONDITIONS"]["educ_min"] = educ_min

    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["low_bound"] = low_bound
    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["middle_bound"] = middle_bound
    model_spec_init_dict["EDUC_LEVEL_BOUNDS"]["high_bound"] = high_bound

    model_spec_init_dict["SIMULATION"]["seed_sim"] = seed_sim
    model_spec_init_dict["SIMULATION"]["num_agents_sim"] = agents

    model_spec_init_dict["SOLUTION"]["seed_emax"] = seed_emax
    model_spec_init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax

    print_dict(model_spec_init_dict)

    # Generate model params init data frame

    model_params_init_dict = dict()

    (
        model_params_init_dict["gamma_0s1"],
        model_params_init_dict["gamma_0s2"],
        model_params_init_dict["gamma_0s3"],
    ) = np.random.uniform(0.5, 4.0, 3).tolist()

    (
        model_params_init_dict["gamma_1s1"],
        model_params_init_dict["gamma_1s2"],
        model_params_init_dict["gamma_1s3"],
    ) = np.random.uniform(0.08, 0.3, 3).tolist()

    (
        model_params_init_dict["g_s1"],
        model_params_init_dict["g_s2"],
        model_params_init_dict["g_s3"],
    ) = np.random.uniform(0.02, 0.5, 3).tolist()

    (
        model_params_init_dict["delta_s1"],
        model_params_init_dict["delta_s2"],
        model_params_init_dict["delta_s3"],
    ) = np.random.uniform(0.1, 0.9, 3).tolist()

    (
        model_params_init_dict["const_p"],
        model_params_init_dict["const_f"],
    ) = np.random.uniform(0.5, 5, 2).tolist()

    # Random number of types: 1, 2, 3, or 4
    num_types = int(np.random.choice([1, 2, 3, 4], 1))
    # Draw shares that sum up to one
    shares = np.random.uniform(1, 10, num_types)
    shares /= shares.sum()
    shares = shares.tolist()

    for i in range(1, num_types):
        # Draw random parameters
        (
            model_params_init_dict["theta_p" + "{}".format(i)],
            model_params_init_dict["theta_f" + "{}".format(i)],
        ) = np.random.uniform(-0.05, -4, 2).tolist()

        # Assign shares
        model_params_init_dict["share_" + "{}".format(i)] = shares[i]

    (
        model_params_init_dict["sigma_1"],
        model_params_init_dict["sigma_2"],
        model_params_init_dict["sigma_3"],
    ) = np.random.uniform(0.002, 2.0, 3).tolist()

    # Determine categories
    category = []

    for (key, _) in model_params_init_dict.items():
        # Check if key is even then add pair to new dictionary
        if "gamma_0" in key:
            category.append("const_wage_eq")
        elif "gamma_1" in key:
            category.append("exp_returns")
        elif "g_s" in key:
            category.append("exp_accm")
        elif "delta" in key:
            category.append("exp_deprec")
        elif "const" in key:
            category.append("disutil_work")
        elif "theta" in key:
            category.append("hetrg_unobs")
        elif "share" in key:
            category.append("shares")
        elif "sigma" in key:
            category.append("sd_wage_shock")

    # Create data frame
    columns = ["name", "value"]

    data = list(model_params_init_dict.items())

    random_model_params_df = pd.DataFrame(data, columns=columns)

    random_model_params_df.insert(0, "category", category, True)

    random_model_params_df.set_index(["category", "name"], inplace=True)

    random_model_params_df.to_pickle("test.soepy.pkl")

    return model_spec_init_dict, random_model_params_df


def print_dict(model_spec_init_dict, file_name="test"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "EDUC_LEVEL_BOUNDS",
        "SIMULATION",
        "SOLUTION",
    ]
    for key_ in order:
        ordered_dict[key_] = model_spec_init_dict[key_]

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

    init_dict["EDUC_LEVEL_BOUNDS"] = dict()
    init_dict["EDUC_LEVEL_BOUNDS"]["low_bound"] = init_dict_flat["low_bound"]
    init_dict["EDUC_LEVEL_BOUNDS"]["middle_bound"] = init_dict_flat["middle_bound"]
    init_dict["EDUC_LEVEL_BOUNDS"]["high_bound"] = init_dict_flat["high_bound"]

    init_dict["SIMULATION"] = dict()
    init_dict["SIMULATION"]["seed_sim"] = init_dict_flat["seed_sim"]
    init_dict["SIMULATION"]["num_agents_sim"] = init_dict_flat["num_agents_sim"]

    init_dict["SOLUTION"] = dict()
    init_dict["SOLUTION"]["seed_emax"] = init_dict_flat["seed_emax"]
    init_dict["SOLUTION"]["num_draws_emax"] = init_dict_flat["num_draws_emax"]

    return init_dict


def read_init_file2(init_file_name):
    """Loads the model specification from yaml file
    as dictionary and expands it without further changes"""

    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.Loader)

    return init_dict
