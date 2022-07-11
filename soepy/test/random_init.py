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
        num_draws_emax = np.random.randint(600, 800)

    if "CHILD_AGE_INIT_MAX" in constr.keys():
        child_age_init_max = constr["CHILD_AGE_INIT_MAX"]
    else:
        child_age_init_max = np.random.randint(0, 4)

    if "INIT_EXP_MAX" in constr.keys():
        init_exp_max = constr["INIT_EXP_MAX"]
    else:
        init_exp_max = np.random.randint(0, 4)

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

    model_spec_init_dict["EDUC"]["educ_years"] = educ_years
    model_spec_init_dict["EXPERIENCE"]["exp_cap"] = exp_cap

    model_spec_init_dict["SIMULATION"]["seed_sim"] = seed_sim
    model_spec_init_dict["SIMULATION"]["num_agents_sim"] = agents

    model_spec_init_dict["SOLUTION"]["seed_emax"] = seed_emax
    model_spec_init_dict["SOLUTION"]["num_draws_emax"] = num_draws_emax
    model_spec_init_dict["SOLUTION"]["integration_method"] = "monte_carlo"

    model_spec_init_dict["TAXES_TRANSFERS"]["alg1_replacement_no_child"] = 0.6
    model_spec_init_dict["TAXES_TRANSFERS"]["alg1_replacement_child"] = 0.67
    model_spec_init_dict["TAXES_TRANSFERS"]["child_benefits"] = 43
    model_spec_init_dict["TAXES_TRANSFERS"]["regelsatz_single"] = 91
    model_spec_init_dict["TAXES_TRANSFERS"]["regelsatz_partner"] = 82
    model_spec_init_dict["TAXES_TRANSFERS"]["regelsatz_child"] = 59
    model_spec_init_dict["TAXES_TRANSFERS"]["motherhood_replacement"] = 0.67
    model_spec_init_dict["TAXES_TRANSFERS"]["elterngeld_min"] = 300
    model_spec_init_dict["TAXES_TRANSFERS"]["elterngeld_max"] = 1800

    model_spec_init_dict["TAXES_TRANSFERS"]["addition_child_single"] = 33
    model_spec_init_dict["TAXES_TRANSFERS"]["housing_single"] = 77.5
    model_spec_init_dict["TAXES_TRANSFERS"]["housing_addtion"] = 15

    model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"] = {}
    model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["under_3"] = [
        219,
        381,
    ]
    model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["3_to_6"] = [
        122,
        128,
    ]
    model_spec_init_dict["TAXES_TRANSFERS"]["ssc_rate"] = 0.215
    model_spec_init_dict["TAXES_TRANSFERS"]["ssc_cap"] = 63_000
    model_spec_init_dict["TAXES_TRANSFERS"]["tax_year"] = 2007
    model_spec_init_dict["TAXES_TRANSFERS"]["tax_splitting"] = True

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
    model_spec_init_dict["INITIAL_CONDITIONS"][
        "ft_exp_shares_file_name"
    ] = "test.soepy.ft.exp.shares.pkl"
    model_spec_init_dict["INITIAL_CONDITIONS"][
        "pt_exp_shares_file_name"
    ] = "test.soepy.pt.exp.shares.pkl"
    model_spec_init_dict["INITIAL_CONDITIONS"]["init_exp_max"] = init_exp_max

    model_spec_init_dict["EXOG_PROC"]["child_info_file_name"] = "test.soepy.child.pkl"
    model_spec_init_dict["EXOG_PROC"][
        "partner_arrival_info_file_name"
    ] = "test.soepy.partner.arrival.pkl"
    model_spec_init_dict["EXOG_PROC"][
        "partner_separation_info_file_name"
    ] = "test.soepy.partner.separation.pkl"
    model_spec_init_dict["EXOG_PROC"]["child_age_max"] = 18
    model_spec_init_dict["EXOG_PROC"]["last_child_bearing_period"] = periods
    model_spec_init_dict["EXOG_PROC"]["partner_cf_const"] = 3
    model_spec_init_dict["EXOG_PROC"]["partner_cf_age"] = 0.3
    model_spec_init_dict["EXOG_PROC"]["partner_cf_age_sq"] = -0.003
    model_spec_init_dict["EXOG_PROC"]["partner_cf_educ"] = 0.03

    print_dict(model_spec_init_dict)

    # Generate model params init data frame

    model_params_init_dict = dict()

    (
        model_params_init_dict["gamma_0_low"],
        model_params_init_dict["gamma_0_middle"],
        model_params_init_dict["gamma_0_high"],
    ) = np.random.uniform(0.5, 4.0, 3).tolist()

    (
        model_params_init_dict["gamma_f_low"],
        model_params_init_dict["gamma_f_middle"],
        model_params_init_dict["gamma_f_high"],
    ) = np.random.uniform(0.001, 0.2, 3).tolist()

    (
        model_params_init_dict["gamma_p_low"],
        model_params_init_dict["gamma_p_middle"],
        model_params_init_dict["gamma_p_high"],
    ) = np.random.uniform(0.001, 0.2, 3).tolist()

    (
        model_params_init_dict["gamma_p_bias_low"],
        model_params_init_dict["gamma_p_bias_middle"],
        model_params_init_dict["gamma_p_bias_high"],
    ) = np.random.uniform(0.6, 1.4, 3).tolist()

    (
        model_params_init_dict["no_kids_f_educ_low"],
        model_params_init_dict["no_kids_f_educ_middle"],
        model_params_init_dict["no_kids_f_educ_high"],
        model_params_init_dict["yes_kids_f_educ_low"],
        model_params_init_dict["yes_kids_f_educ_middle"],
        model_params_init_dict["yes_kids_f_educ_high"],
        model_params_init_dict["child_0_2_f"],
        model_params_init_dict["child_3_5_f"],
        model_params_init_dict["child_6_10_f"],
        model_params_init_dict["child_11_age_max_f"],
    ) = np.random.uniform(0.001, 0.2, 10).tolist()

    (
        model_params_init_dict["no_kids_p_educ_low"],
        model_params_init_dict["no_kids_p_educ_middle"],
        model_params_init_dict["no_kids_p_educ_high"],
        model_params_init_dict["yes_kids_p_educ_low"],
        model_params_init_dict["yes_kids_p_educ_middle"],
        model_params_init_dict["yes_kids_p_educ_high"],
        model_params_init_dict["child_0_2_p"],
        model_params_init_dict["child_3_5_p"],
        model_params_init_dict["child_6_10_p"],
        model_params_init_dict["child_11_age_max_p"],
    ) = np.random.uniform(-1.5, -0.001, 10).tolist()

    model_params_init_dict["delta"] = np.random.uniform(0.8, 0.99)
    model_params_init_dict["mu"] = np.random.uniform(-0.7, -0.4)

    # Random number of types: 1, 2, 3, or 4
    num_types = int(np.random.choice([1, 2, 3, 4], 1))
    # Draw shares that sum up to one
    shares = np.random.uniform(1, 10, num_types)
    shares /= shares.sum()
    shares = shares.tolist()

    for i in range(1, num_types):
        # Draw random parameters
        (
            model_params_init_dict["theta_p" + f"{i}"],
            model_params_init_dict["theta_f" + f"{i}"],
        ) = np.random.uniform(-0.1, -1, 2).tolist()

        # Assign shares
        model_params_init_dict["share_" + f"{i}"] = shares[i]

    model_params_init_dict["sigma"] = np.random.uniform(0.001, 1.0, 1)[0]

    # Determine categories
    category = []

    for (key, _) in model_params_init_dict.items():
        # Check if key is even then add pair to new dictionary
        if "gamma_0" in key:
            category.append("const_wage_eq")
        elif "gamma_f" in key:
            category.append("exp_returns_f")
        elif ("gamma_p" in key) and ("gamma_p_bias" not in key):
            category.append("exp_returns_p")
        elif "gamma_p_bias" in key:
            category.append("exp_returns_p_bias")
        elif "theta" in key:
            category.append("hetrg_unobs")
        elif "share" in key:
            category.append("shares")
        elif "sigma" in key:
            category.append("sd_wage_shock")
        elif "delta" == key:
            category.append("discount")
        elif "mu" == key:
            category.append("risk")
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

    # Generate random fractions for initial exp levels
    for label in ["pt", "ft"]:
        # Constrained model with initial experience 0
        if init_exp_max == 0:
            exper_shares = np.repeat(0.00, len(educ_years))
            index_levels = [list(range(len(educ_years))), [init_exp_max]]
        # Individuals may have worked before entering the model
        else:
            exper_shares = np.random.uniform(
                1, 10, size=(init_exp_max + 1) * len(educ_years)
            )
            exper_shares /= exper_shares.sum()
            index_levels = [[0, 1, 2], list(range(0, init_exp_max + 1))]
        index = pd.MultiIndex.from_product(
            index_levels, names=["educ_level", label + "_exp"]
        )
        exog_exper_shares = pd.DataFrame(
            exper_shares.tolist(),
            index=index,
            columns=["exper_shares"],
        )
        exog_exper_shares.to_pickle("test.soepy." + label + ".exp.shares.pkl")
        if label == "pt":
            exog_exper_shares_pt = exog_exper_shares
        else:
            exog_exper_shares_ft = exog_exper_shares

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
        exog_exper_shares_pt,
        exog_exper_shares_ft,
        exog_child_info,
        exog_partner_arrival_info,
        exog_partner_separation_info,
    )


def print_dict(model_spec_init_dict, file_name="test"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
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

    with open(f"{file_name}.soepy.yml", "w") as outfile:
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
    init_dict["TAXES_TRANSFERS"]["alg1_replacement_no_child"] = init_dict_flat[
        "alg1_replacement_no_child"
    ]
    init_dict["TAXES_TRANSFERS"]["alg1_replacement_child"] = init_dict_flat[
        "alg1_replacement_child"
    ]
    init_dict["TAXES_TRANSFERS"]["child_benefits"] = init_dict_flat["child_benefits"]
    init_dict["TAXES_TRANSFERS"]["regelsatz_single"] = init_dict_flat[
        "regelsatz_single"
    ]
    init_dict["TAXES_TRANSFERS"]["regelsatz_partner"] = init_dict_flat[
        "regelsatz_partner"
    ]
    init_dict["TAXES_TRANSFERS"]["regelsatz_child"] = init_dict_flat["regelsatz_child"]
    init_dict["TAXES_TRANSFERS"]["motherhood_replacement"] = init_dict_flat[
        "motherhood_replacement"
    ]
    init_dict["TAXES_TRANSFERS"]["elterngeld_min"] = init_dict_flat["elterngeld_min"]
    init_dict["TAXES_TRANSFERS"]["elterngeld_max"] = init_dict_flat["elterngeld_max"]

    init_dict["TAXES_TRANSFERS"]["addition_child_single"] = init_dict_flat[
        "addition_child_single"
    ]
    init_dict["TAXES_TRANSFERS"]["housing_single"] = init_dict_flat["housing_single"]
    init_dict["TAXES_TRANSFERS"]["housing_addtion"] = init_dict_flat["housing_addtion"]

    init_dict["TAXES_TRANSFERS"]["ssc_rate"] = init_dict_flat["ssc_rate"]
    init_dict["TAXES_TRANSFERS"]["ssc_cap"] = init_dict_flat["ssc_cap"]

    init_dict["TAXES_TRANSFERS"]["tax_year"] = init_dict_flat["tax_year"]

    init_dict["INITIAL_CONDITIONS"] = dict()
    init_dict["INITIAL_CONDITIONS"]["child_age_init_max"] = init_dict_flat[
        "child_age_init_max"
    ]
    init_dict["INITIAL_CONDITIONS"]["init_exp_max"] = init_dict_flat["init_exp_max"]

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
