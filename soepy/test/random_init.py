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

    if "NUM_EDUC_LEVELS" in constr.keys():
        num_educ_levels = constr["NUM_EDUC_LEVELS"]
    else:
        num_educ_levels = 3

    if "EDUC_YEARS_EDUC_LEVEL_LOW" in constr.keys():
        educ_years_educ_level_low = constr["EDUC_YEARS_EDUC_LEVEL_LOW"]
    else:
        educ_years_educ_level_low = 0

    if "EDUC_YEARS_EDUC_LEVEL_MIDDLE" in constr.keys():
        educ_years_educ_level_middle = constr["EDUC_YEARS_EDUC_LEVEL_MIDDLE"]
    else:
        educ_years_educ_level_middle = 2

    if "EDUC_YEARS_EDUC_LEVEL_HIGH" in constr.keys():
        educ_years_educ_level_high = constr["EDUC_YEARS_EDUC_LEVEL_HIGH"]
    else:
        educ_years_educ_level_high = 6

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

    if "BENEFITS_BASE" in constr.keys():
        benefits_base = constr["BENEFITS_BASE"]
    else:
        benefits_base = np.random.uniform(0, 500)

    if "BENEFITS_KIDS" in constr.keys():
        benefits_kids = constr["BENEFITS_KIDS"]
    else:
        benefits_kids = np.random.uniform(0, 100)

    model_spec_init_dict = dict()

    for key_ in [
        "GENERAL",
        "CONSTANTS",
        "EDUC",
        "SIMULATION",
        "SOLUTION",
        "TAXES_TRANSFERS",
        "EXOG_PROC",
    ]:
        model_spec_init_dict[key_] = {}

    model_spec_init_dict["GENERAL"]["num_periods"] = periods

    model_spec_init_dict["CONSTANTS"]["delta"] = np.random.uniform(0.8, 0.99)
    model_spec_init_dict["CONSTANTS"]["mu"] = np.random.uniform(-0.7, -0.4)

    model_spec_init_dict["EDUC"]["num_educ_levels"] = num_educ_levels
    model_spec_init_dict["EDUC"][
        "educ_years_educ_level_low"
    ] = educ_years_educ_level_low
    model_spec_init_dict["EDUC"][
        "educ_years_educ_level_middle"
    ] = educ_years_educ_level_middle
    model_spec_init_dict["EDUC"][
        "educ_years_educ_level_high"
    ] = educ_years_educ_level_high

    model_spec_init_dict["SIMULATION"]["seed_sim"] = seed_sim
    model_spec_init_dict["SIMULATION"]["num_agents_sim"] = agents

    model_spec_init_dict["SOLUTION"]["seed_emax"] = seed_emax
    model_spec_init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax

    model_spec_init_dict["TAXES_TRANSFERS"]["benefits_base"] = benefits_base
    model_spec_init_dict["TAXES_TRANSFERS"]["benefits_kids"] = benefits_kids

    model_spec_init_dict["EXOG_PROC"]["educ_info_file_name"] = "test.soepy.educ.pkl"
    model_spec_init_dict["EXOG_PROC"]["kids_info_file_name"] = "test.soepy.child.pkl"
    model_spec_init_dict["EXOG_PROC"][
        "partner_info_file_name"
    ] = "test.soepy.partner.pkl"
    model_spec_init_dict["EXOG_PROC"]["child_age_max"] = 12
    model_spec_init_dict["EXOG_PROC"]["last_child_bearing_period"] = periods
    model_spec_init_dict["EXOG_PROC"]["partner_cf_const"] = 3
    model_spec_init_dict["EXOG_PROC"]["partner_cf_age"] = 0.3
    model_spec_init_dict["EXOG_PROC"]["partner_cf_age_sq"] = -0.003
    model_spec_init_dict["EXOG_PROC"]["partner_cf_educ"] = 0.03

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
        model_params_init_dict["no_kids_f"],
        model_params_init_dict["no_kids_p"],
        model_params_init_dict["yes_kids_f"],
        model_params_init_dict["yes_kids_p"],
        model_params_init_dict["child_02_f"],
        model_params_init_dict["child_02_p"],
        model_params_init_dict["child_35_f"],
        model_params_init_dict["child_35_p"],
        model_params_init_dict["child_610_f"],
        model_params_init_dict["child_610_p"],
    ) = np.random.uniform(0.5, 5, 10).tolist()

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
        elif "theta" in key:
            category.append("hetrg_unobs")
        elif "share" in key:
            category.append("shares")
        elif "sigma" in key:
            category.append("sd_wage_shock")
        elif "kids" or "child" in key:
            category.append("disutil_work")

    # Create data frame
    columns = ["name", "value"]

    data = list(model_params_init_dict.items())

    random_model_params_df = pd.DataFrame(data, columns=columns)

    random_model_params_df.insert(0, "category", category, True)

    random_model_params_df.set_index(["category", "name"], inplace=True)

    random_model_params_df.to_pickle("test.soepy.pkl")

    # Generate random probabilities of childbirth
    exog_child_info = pd.DataFrame(
        np.random.uniform(0, 1, size=periods).tolist(),
        index=list(range(0, periods)),
        columns=["prob_child_values"],
    )
    exog_child_info.to_pickle("test.soepy.child.pkl")

    # Generate random probabilities of marriage
    index_levels = [list(range(0, periods)), [0, 1, 2]]

    index = pd.MultiIndex.from_product(index_levels, names=["period", "educ_level"])
    exog_partner_info = pd.DataFrame(
        np.zeros(periods * 3).tolist(), index=index, columns=["exog_partner_values"]
    )
    exog_partner_info.to_pickle("test.soepy.partner.pkl")

    # Generate random fractions for education levels
    educ_shares = np.random.uniform(1, 10, size=num_educ_levels)
    educ_shares /= educ_shares.sum()
    exog_educ_info = pd.DataFrame(
        educ_shares.tolist(),
        index=list(range(0, num_educ_levels)),
        columns=["Fraction"],
    )
    exog_educ_info.to_pickle("test.soepy.educ.pkl")

    # Generate random probabilities of partner arrival
    index_levels = [list(range(0, periods)), [0, 1, 2]]
    index = pd.MultiIndex.from_product(index_levels, names=["period", "educ_level"])
    if "PARTNER" in constr.keys():
        exog_partner_info = pd.DataFrame(
            np.zeros(periods * 3).tolist(),
            index=index,
            columns=["exog_partner_values"],
        )
    else:
        exog_partner_info = pd.DataFrame(
            np.random.uniform(0, 1, size=periods * 3).tolist(),
            index=index,
            columns=["exog_partner_values"],
        )

    exog_partner_info.to_pickle("test.soepy.partner.pkl")

    return (
        model_spec_init_dict,
        random_model_params_df,
        exog_child_info,
        exog_educ_info,
        exog_partner_info,
    )


def print_dict(model_spec_init_dict, file_name="test"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "EDUC",
        "SIMULATION",
        "SOLUTION",
        "TAXES_TRANSFERS",
        "EXOG_PROC",
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

    init_dict["EDUC"] = dict()
    init_dict["EDUC"]["num_educ_levels"] = init_dict_flat["num_educ_levels"]
    init_dict["EDUC"]["educ_years_educ_level_low"] = init_dict_flat[
        "educ_years_educ_level_low"
    ]
    init_dict["EDUC"]["educ_years_educ_level_middle"] = init_dict_flat[
        "educ_years_educ_level_middle"
    ]
    init_dict["EDUC"]["educ_years_educ_level_high"] = init_dict_flat[
        "educ_years_educ_level_high"
    ]

    init_dict["SIMULATION"] = dict()
    init_dict["SIMULATION"]["seed_sim"] = init_dict_flat["seed_sim"]
    init_dict["SIMULATION"]["num_agents_sim"] = init_dict_flat["num_agents_sim"]

    init_dict["SOLUTION"] = dict()
    init_dict["SOLUTION"]["seed_emax"] = init_dict_flat["seed_emax"]
    init_dict["SOLUTION"]["num_draws_emax"] = init_dict_flat["num_draws_emax"]

    init_dict["TAXES_TRANSFERS"] = dict()
    init_dict["TAXES_TRANSFERS"]["benefits_base"] = init_dict_flat["benefits_base"]
    init_dict["TAXES_TRANSFERS"]["benefits_kids"] = init_dict_flat["benefits_kids"]

    init_dict["EXOG_PROC"] = dict()
    init_dict["EXOG_PROC"]["kids_info_file_name"] = init_dict_flat[
        "kids_info_file_name"
    ]
    init_dict["EXOG_PROC"]["child_age_max"] = init_dict_flat["child_age_max"]
    init_dict["EXOG_PROC"]["last_child_bearing_period"] = init_dict_flat[
        "last_child_bearing_period"
    ]

    return init_dict


def read_init_file2(init_file_name):
    """Loads the model specification from yaml file
    as dictionary and expands it without further changes"""

    # Import yaml initialization file as dictionary init_dict
    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.Loader)

    return init_dict
