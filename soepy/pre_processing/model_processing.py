import collections
import copy

import pandas as pd
import yaml

from soepy.pre_processing.tax_and_transfers_params import create_child_care_costs
from soepy.pre_processing.tax_and_transfers_params import process_elterngeld
from soepy.pre_processing.tax_and_transfers_params import process_ssc
from soepy.pre_processing.tax_and_transfers_params import process_tax_system


def read_model_params_init(model_params_init_file_name):
    """Reads in specification of model parameters
    from a pickled data frame and saves parameters as named tuple."""

    if isinstance(model_params_init_file_name, str):
        model_params_df = pd.read_pickle(model_params_init_file_name)
    elif isinstance(model_params_init_file_name, pd.DataFrame):
        model_params_df = model_params_init_file_name
    else:
        raise NotImplementedError

    # Transform data frame to dictionary
    model_params_dict = {
        l: model_params_df.loc[l, "value"].to_dict()
        for l in model_params_df.index.levels[0]
    }

    # Add share of baseline type to parameters dictionary
    model_params_dict_expanded = expand_model_params_dict(model_params_dict)

    # Remove nested structure of dictionary
    model_params_dict_flat = group_parameters(model_params_dict_expanded)

    # Save as namedtuple
    model_params = dict_to_namedtuple_params(model_params_dict_flat)

    return model_params_df, model_params


def expand_model_params_dict(model_params_dict):
    # Calculate covariances of the error terms given standard deviations
    shocks_cov = [
        model_params_dict["sd_wage_shock"]["sigma_1"],
        model_params_dict["sd_wage_shock"]["sigma_2"],
    ]
    shocks_cov = [shocks_cov[0] ** 2, shocks_cov[1] ** 2]

    # Extract the values of the type shares
    try:
        type_shares_non_baseline = [
            _ for k, _ in model_params_dict["shares"].items() if "share" in k
        ]

        num_types = len(type_shares_non_baseline) + 1

        # Aggregate type shares in list object
        # Share of baseline types equal to one minus sum of remaining type shares
        type_shares = [1 - sum(type_shares_non_baseline)] + type_shares_non_baseline

    except KeyError:

        type_shares = [1]
        num_types = 1

    # Append derived attributes to init_dict
    model_params_dict["derived_attr"] = {
        "shocks_cov": shocks_cov,
        "type_shares": type_shares,
        "num_types": num_types,
    }

    return model_params_dict


def group_parameters(model_params_dict_expanded):
    """Groups the parameters to be estimates
    in flat dictionary structure"""

    model_params_dict_flat = dict()

    model_params_dict_flat["gamma_0s"] = list(
        model_params_dict_expanded["const_wage_eq"].values()
    )

    model_params_dict_flat["gamma_1s"] = list(
        model_params_dict_expanded["exp_returns"].values()
    )

    model_params_dict_flat["g_s"] = list(
        model_params_dict_expanded["exp_accm"].values()
    )

    model_params_dict_flat["g_bar_s"] = list(
        model_params_dict_expanded["exp_accm_expected"].values()
    )

    model_params_dict_flat["delta_s"] = list(
        model_params_dict_expanded["exp_deprec"].values()
    )

    for key_ in list(model_params_dict_expanded["disutil_work"].keys()):
        if "child" in key_:
            model_params_dict_flat[key_] = model_params_dict_expanded["disutil_work"][
                key_
            ]

    for i in ["no", "yes"]:
        for j in ["f", "p"]:

            model_params_dict_flat[i + "_kids_" + j] = [
                model_params_dict_expanded["disutil_work"][
                    i + "_kids_" + j + "_educ_low"
                ],
                model_params_dict_expanded["disutil_work"][
                    i + "_kids_" + j + "_educ_middle"
                ],
                model_params_dict_expanded["disutil_work"][
                    i + "_kids_" + j + "_educ_high"
                ],
            ]

    model_params_dict_flat["shocks_cov"] = model_params_dict_expanded["derived_attr"][
        "shocks_cov"
    ]
    model_params_dict_flat["type_shares"] = model_params_dict_expanded["derived_attr"][
        "type_shares"
    ]

    if model_params_dict_expanded["derived_attr"]["num_types"] > 1:
        for i in ["p", "f"]:
            model_params_dict_flat["theta_" + i] = [
                v
                for k, v in model_params_dict_expanded["hetrg_unobs"].items()
                if "{}".format("theta_" + i) in k
            ]

    else:
        pass

    return model_params_dict_flat


def dict_to_namedtuple_params(dictionary):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple("model_parameters", dictionary.keys())(**dictionary)


def read_model_spec_init(model_spec_init_dict, model_params):
    """Reads in the model specification from yaml file.
    This initialisation component contains only information
    that does not change during estimation. Inputs are made
    available as named tuple."""

    # Import yaml initialization file as dictionary init_dict
    if isinstance(model_spec_init_dict, str):
        with open(model_spec_init_dict) as y:
            model_spec_init = yaml.load(y, Loader=yaml.Loader)
    else:
        model_spec_init = copy.deepcopy(model_spec_init_dict)

    model_spec_dict_expand = expand_model_spec_dict(model_spec_init, model_params)

    model_spec_dict_flat = flatten_model_spec_dict(model_spec_dict_expand)

    model_spec = dict_to_namedtuple_spec(model_spec_dict_flat)

    return model_spec


def expand_model_spec_dict(model_spec_init_dict, model_params_df):
    # Gather education years in list object
    num_educ_levels = len(model_spec_init_dict["EDUC"]["educ_years"])

    # Determine number of types
    try:
        num_types = len(model_params_df.loc["shares"].to_numpy()) + 1
    except KeyError:
        num_types = 1

    # Append derived attributes to init_dict
    model_spec_init_dict["DERIVED_ATTR"] = {
        "num_educ_levels": num_educ_levels,
        "num_types": num_types,
    }

    model_spec_init_dict = process_tax_system(model_spec_init_dict)
    model_spec_init_dict = create_child_care_costs(model_spec_init_dict)
    model_spec_init_dict = process_ssc(model_spec_init_dict)
    model_spec_init_dict = process_elterngeld(model_spec_init_dict)

    return model_spec_init_dict


def flatten_model_spec_dict(model_spec_dict):
    """Removes the grouping from the nested dictionary"""

    groups = [
        "GENERAL",
        "CONSTANTS",
        "EDUC",
        "EXPERIENCE",
        "SIMULATION",
        "SOLUTION",
        "TAXES_TRANSFERS",
        "EXOG_PROC",
        "INITIAL_CONDITIONS",
        "DERIVED_ATTR",
    ]

    model_spec_dict_flat = dict()

    for group in groups:

        keys_ = list(model_spec_dict[group].keys())
        values_ = list(model_spec_dict[group].values())

        for k_, key_ in enumerate(keys_):
            model_spec_dict_flat[key_] = values_[k_]

    return model_spec_dict_flat


def dict_to_namedtuple_spec(dictionary):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple("model_specification", dictionary.keys())(
        **dictionary
    )
