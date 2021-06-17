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

    if "EDUC_YEARS" in constr.keys():
        educ_years = constr["EDUC_YEARS"]
    else:
        educ_years = [0, 2, 6]

    if "EXPERIENCE" in constr.keys():
        exp_cap = constr["EXPERIENCE"]
    else:
        exp_cap = 15

    if "AGENTS" in constr.keys():
        agents = constr["AGENTS"]
    else:
        agents = np.random.randint(500, 1000)

    if "PERIODS" in constr.keys():
        periods = constr["PERIODS"]
    else:
        periods = np.random.randint(3, 8)

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

    if "CHILD_AGE_INIT_MAX" in constr.keys():
        child_age_init_max = constr["CHILD_AGE_INIT_MAX"]
    else:
        child_age_init_max = np.random.randint(0, 4)

    model_spec_init_dict = dict()

    for key_ in [
        "GENERAL",
        "CONSTANTS",
        "EDUC",
        "EXPERIENCE",
        "SIMULATION",
        "SOLUTION",
        "TAXES_TRANSFERS",
        "INITIAL_CONDITIONS",
        "EXOG_PROC",
    ]:
        model_spec_init_dict[key_] = {}

    model_spec_init_dict["GENERAL"]["num_periods"] = periods

    model_spec_init_dict["CONSTANTS"]["delta"] = np.random.uniform(0.8, 0.99)
    model_spec_init_dict["CONSTANTS"]["mu"] = np.random.uniform(-0.7, -0.4)

    model_spec_init_dict["EDUC"]["educ_years"] = educ_years
    model_spec_init_dict["EXPERIENCE"]["exp_cap"] = exp_cap

    model_spec_init_dict["SIMULATION"]["seed_sim"] = seed_sim
    model_spec_init_dict["SIMULATION"]["num_agents_sim"] = agents

    model_spec_init_dict["SOLUTION"]["seed_emax"] = seed_emax
    model_spec_init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax

    model_spec_init_dict["TAXES_TRANSFERS"]["benefits_base"] = benefits_base
    model_spec_init_dict["TAXES_TRANSFERS"]["benefits_kids"] = benefits_kids

    model_spec_init_dict["INITIAL_CONDITIONS"][
        "educ_shares_file_name"
    ] = "test.soepy.educ.shares.pkl"
    model_spec_init_dict["INITIAL_CONDITIONS"][
        "child_age_shares_file_name"
    ] = "test.soepy.child.age.shares.pkl"
    model_spec_init_dict["INITIAL_CONDITIONS"][
        "child_age_init_max"
    ] = child_age_init_max
    model_spec_init_dict["INITIAL_CONDITIONS"][
        "partner_shares_file_name"
    ] = "test.soepy.partner.shares.pkl"

    model_spec_init_dict["EXOG_PROC"]["child_info_file_name"] = "test.soepy.child.pkl"
    model_spec_init_dict["EXOG_PROC"][
        "partner_arrival_info_file_name"
    ] = "test.soepy.partner.arrival.pkl"
    model_spec_init_dict["EXOG_PROC"][
        "partner_separation_info_file_name"
    ] = "test.soepy.partner.separation.pkl"
    model_spec_init_dict["EXOG_PROC"]["child_age_max"] = 10
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
    ) = np.random.uniform(0.001, 0.5, 3).tolist()

    (
        model_params_init_dict["g_s1"],
        model_params_init_dict["g_s2"],
        model_params_init_dict["g_s3"],
    ) = np.random.uniform(0.001, 0.5, 3).tolist()

    (
        model_params_init_dict["g_bar_s1"],
        model_params_init_dict["g_bar_s2"],
        model_params_init_dict["g_bar_s3"],
    ) = [0.570, 0.533, 0.625]

    (
        model_params_init_dict["delta_s1"],
        model_params_init_dict["delta_s2"],
        model_params_init_dict["delta_s3"],
    ) = np.random.uniform(0.001, 0.2, 3).tolist()

    (
        model_params_init_dict["no_kids_f_educ_low"],
        model_params_init_dict["no_kids_f_educ_middle"],
        model_params_init_dict["no_kids_f_educ_high"],
        model_params_init_dict["yes_kids_f_educ_low"],
        model_params_init_dict["yes_kids_f_educ_middle"],
        model_params_init_dict["yes_kids_f_educ_high"],
        model_params_init_dict["child_02_f"],
        model_params_init_dict["child_35_f"],
        model_params_init_dict["child_6orolder_f"],
    ) = np.random.uniform(0.001, 0.2, 9).tolist()

    (
        model_params_init_dict["no_kids_p_educ_low"],
        model_params_init_dict["no_kids_p_educ_middle"],
        model_params_init_dict["no_kids_p_educ_high"],
        model_params_init_dict["yes_kids_p_educ_low"],
        model_params_init_dict["yes_kids_p_educ_middle"],
        model_params_init_dict["yes_kids_p_educ_high"],
        model_params_init_dict["child_02_p"],
        model_params_init_dict["child_35_p"],
        model_params_init_dict["child_6orolder_p"],
    ) = np.random.uniform(-1.5, -0.001, 9).tolist()

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
        ) = np.random.uniform(-0.1, -1, 2).tolist()

        # Assign shares
        model_params_init_dict["share_" + "{}".format(i)] = shares[i]

    (
        model_params_init_dict["sigma_1"],
        model_params_init_dict["sigma_2"],
    ) = np.random.uniform(0.001, 1.0, 2).tolist()

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
        elif "g_bar_s" in key:
            category.append("exp_accm_expected")
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

    # Random initial conditions
    # Generate random fractions for education levels
    educ_shares = np.random.uniform(1, 10, size=len(educ_years))
    educ_shares /= educ_shares.sum()
    exog_educ_shares = pd.DataFrame(
        educ_shares.tolist(),
        index=list(range(0, len(educ_years))),
        columns=["educ_shares"],
    )
    exog_educ_shares.index.name = "educ_level"
    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")

    # Generate random fractions for initial child ages
    # Constrained model without kids
    if child_age_init_max == -1:
        child_age_shares = np.repeat(0.00, len(educ_years))
        index_levels = [list(range(len(educ_years))), [child_age_init_max]]
    # Kids are part of the model
    else:
        child_age_shares = np.random.uniform(
            1, 10, size=(child_age_init_max + 2) * len(educ_years)
        )
        child_age_shares /= child_age_shares.sum()
        index_levels = [[0, 1, 2], list(range(-1, child_age_init_max + 1))]
    index = pd.MultiIndex.from_product(index_levels, names=["educ_level", "child_age"])
    exog_child_age_shares = pd.DataFrame(
        child_age_shares.tolist(),
        index=index,
        columns=["child_age_shares"],
    )
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")

    # Generate random fractions for partner present in initial period
    exog_partner_shares = pd.DataFrame(
        np.random.uniform(0, 1, size=len(educ_years)).tolist(),
        index=[0, 1, 2],
        columns=["partner_shares"],
    )
    exog_partner_shares.index.name = "educ_level"
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")

    # Random process evolution throughout the model
    # Generate random probabilities of childbirth
    index_levels = [list(range(0, periods)), [0, 1, 2]]
    index = pd.MultiIndex.from_product(index_levels, names=["period", "educ_level"])

    exog_child_info = pd.DataFrame(
        np.random.uniform(0, 1, size=periods * 3).tolist(),
        index=index,
        columns=["prob_child_values"],
    )
    exog_child_info.to_pickle("test.soepy.child.pkl")

    # Generate random probabilities of partner arrival
    if "PARTNER_ARRIVAL" in constr.keys():
        exog_partner_arrival_info = pd.DataFrame(
            np.zeros(periods * 3).tolist(),
            index=index,
            columns=["prob_partner_values"],
        )
    else:
        exog_partner_arrival_info = pd.DataFrame(
            np.random.uniform(0, 1, size=periods * 3).tolist(),
            index=index,
            columns=["prob_partner_values"],
        )

    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")

    # Generate random probabilities of partner separation
    if "PARTNER_SEPARATION" in constr.keys():
        exog_partner_separation_info = pd.DataFrame(
            np.zeros(periods * 3).tolist(),
            index=index,
            columns=["prob_partner_values"],
        )
    else:
        exog_partner_separation_info = pd.DataFrame(
            np.random.uniform(0, 1, size=periods * 3).tolist(),
            index=index,
            columns=["prob_partner_values"],
        )

    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    return (
        model_spec_init_dict,
        random_model_params_df,
        exog_educ_shares,
        exog_child_age_shares,
        exog_partner_shares,
        exog_child_info,
        exog_partner_arrival_info,
        exog_partner_separation_info,
    )


def print_dict(model_spec_init_dict, file_name="test"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "EDUC",
        "EXPERIENCE",
        "SIMULATION",
        "SOLUTION",
        "TAXES_TRANSFERS",
        "INITIAL_CONDITIONS",
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
    init_dict["EDUC"]["educ_years"] = init_dict_flat["educ_years"]
    init_dict["EXPERIENCE"] = dict()
    init_dict["EXPERIENCE"]["exp_cap"] = init_dict_flat["exp_cap"]

    init_dict["SIMULATION"] = dict()
    init_dict["SIMULATION"]["seed_sim"] = init_dict_flat["seed_sim"]
    init_dict["SIMULATION"]["num_agents_sim"] = init_dict_flat["num_agents_sim"]

    init_dict["SOLUTION"] = dict()
    init_dict["SOLUTION"]["seed_emax"] = init_dict_flat["seed_emax"]
    init_dict["SOLUTION"]["num_draws_emax"] = init_dict_flat["num_draws_emax"]

    init_dict["TAXES_TRANSFERS"] = dict()
    init_dict["TAXES_TRANSFERS"]["benefits_base"] = init_dict_flat["benefits_base"]
    init_dict["TAXES_TRANSFERS"]["benefits_kids"] = init_dict_flat["benefits_kids"]

    init_dict["INITIAL_CONDITIONS"] = dict()
    init_dict["INITIAL_CONDITIONS"]["child_age_init_max"] = init_dict_flat[
        "child_age_init_max"
    ]

    init_dict["EXOG_PROC"] = dict()
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
