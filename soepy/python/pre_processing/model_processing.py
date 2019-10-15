import collections

import yaml

import pandas as pd


def transform_old_init_dict_to_df(init_file_name):
    """Reads in init file in yaml format or
    dictionary as in soepy master branch.
    Transforms dictionary in a parameters data frame
    we wish to establish as the new init file format."""

    # Import yaml initialization file as dictionary init_dict
    if isinstance(init_file_name, str):
        with open(init_file_name) as y:
            init_dict = yaml.load(y, Loader=yaml.FullLoader)
    else:
        init_dict = init_file_name

    # Determine categories
    category = []

    for (key, value) in init_dict["PARAMETERS"].items():
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

    data = list(init_dict["PARAMETERS"].items())

    model_params_df = pd.DataFrame(data, columns=columns)

    model_params_df.insert(0, "category", category, True)

    model_params_df.set_index(["category", "name"], inplace=True)

    return model_params_df


# Pre-processing of model parameters to be estimated


def read_model_params_init(model_params_df):
    """Reads in specification of model parameters
    from a pickled data frame and saves parameters as named tuple."""

    # Transform data frame to dictionary
    model_params_dict = {
        l: model_params_df.xs(l)["value"].to_dict()
        for l in model_params_df.index.levels[0]
    }

    # Add share of baseline type to parameters dictionary
    model_params_dict_expanded = expand_model_params_dict(model_params_dict)

    # Remove nested structure of dictionary
    model_params_dict_flat = flatten_model_params_dict(model_params_dict_expanded)

    # Save as namedtuple
    model_params = dict_to_namedtuple_params(model_params_dict_flat)

    return model_params


def expand_model_params_dict(model_params_dict):
    # Extract the values of the type shares
    type_shares_non_baseline = [
        _ for k, _ in model_params_dict["shares"].items() if "share" in k
    ]

    # Share of baseline types equal to one minus sum of remaining type shares
    share_0 = 1 - sum(type_shares_non_baseline)

    # Append derived attribute to model_params_dict
    model_params_dict["shares"].update({"share_0": share_0})

    return model_params_dict


def flatten_model_params_dict(model_params_dict):
    """Removes the grouping from the nested dictionary"""

    groups = [
        "const_wage_eq",
        "disutil_work",
        "exp_accm",
        "exp_deprec",
        "exp_returns",
        "hetrg_unobs",
        "sd_wage_shock",
        "shares",
    ]

    model_params_dict_flat = dict()

    for group in groups:

        keys_ = list(model_params_dict[group].keys())
        values_ = list(model_params_dict[group].values())

        for k_, key_ in enumerate(keys_):
            model_params_dict_flat[key_] = values_[k_]

    return model_params_dict_flat


def dict_to_namedtuple_params(dictionary):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple("model_parameters", dictionary.keys())(**dictionary)


# Pre-processing of model specification: values that do not change during estimation


def read_model_spec_init(model_spec_init, model_params):
    """Reads in the model specification from yaml file.
    This initialisation component contains only information
    that does not change dirung estimation. Inputs are made
    available as named tuple."""

    # Import yaml initialization file as dictionary init_dict
    if isinstance(model_spec_init, str):
        with open(model_spec_init) as y:
            model_spec_init_dict = yaml.load(y, Loader=yaml.FullLoader)
    else:
        model_spec_init_dict = model_spec_init

    model_spec_dict = expand_model_spec_dict(model_spec_init_dict, model_params)

    model_spec_dict_flat = flatten_model_spec_dict(model_spec_dict)

    model_spec = dict_to_namedtuple_spec(model_spec_dict_flat)

    return model_spec


def expand_model_spec_dict(model_spec_init_dict, model_params_df):
    # Calculate range of years of education in the (simulated) sample
    educ_min = model_spec_init_dict["INITIAL_CONDITIONS"]["educ_min"]
    educ_max = model_spec_init_dict["INITIAL_CONDITIONS"]["educ_max"]
    educ_range = educ_max - educ_min + 1

    # Determine number of types
    num_types = len(model_params_df.loc["shares"].to_numpy()) + 1

    # Append derived attributes to init_dict
    model_spec_init_dict["DERIVED_ATTR"] = {
        "educ_range": educ_range,
        "num_types": num_types,
    }

    return model_spec_init_dict


def flatten_model_spec_dict(model_spec_dict):
    """Removes the grouping from the nested dictionary"""

    groups = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
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


def group_parameters(init_dict, init_dict_flat):
    """Groups the parameters to be estimates
    in flat dictionary structure"""

    init_dict_flat["gamma_0s"] = (
        init_dict["PARAMETERS"]["gamma_0s1"],
        init_dict["PARAMETERS"]["gamma_0s2"],
        init_dict["PARAMETERS"]["gamma_0s3"],
    )

    init_dict_flat["gamma_1s"] = (
        init_dict["PARAMETERS"]["gamma_1s1"],
        init_dict["PARAMETERS"]["gamma_1s2"],
        init_dict["PARAMETERS"]["gamma_1s3"],
    )

    init_dict_flat["g_s"] = (
        init_dict["PARAMETERS"]["g_s1"],
        init_dict["PARAMETERS"]["g_s2"],
        init_dict["PARAMETERS"]["g_s3"],
    )

    init_dict_flat["delta_s"] = (
        init_dict["PARAMETERS"]["delta_s1"],
        init_dict["PARAMETERS"]["delta_s2"],
        init_dict["PARAMETERS"]["delta_s3"],
    )

    if init_dict["DERIVED_ATTR"]["num_types"] > 1:
        for i in ["p", "f"]:
            init_dict_flat["theta_" + i] = [
                v
                for k, v in init_dict["PARAMETERS"].items()
                if "{}".format("theta_" + i) in k
            ]

        for i in range(1, init_dict["DERIVED_ATTR"]["num_types"]):
            init_dict_flat["share_" + "{}".format(i)] = init_dict["PARAMETERS"][
                "share_" + "{}".format(i)
            ]
    else:
        pass

    init_dict_flat["gamma_1s"] = (
        init_dict["PARAMETERS"]["gamma_1s1"],
        init_dict["PARAMETERS"]["gamma_1s2"],
        init_dict["PARAMETERS"]["gamma_1s3"],
    )

    init_dict_flat["sigma"] = (
        init_dict["PARAMETERS"]["sigma_1"],
        init_dict["PARAMETERS"]["sigma_2"],
        init_dict["PARAMETERS"]["sigma_3"],
    )

    init_dict_flat["const_p"] = init_dict["PARAMETERS"]["const_p"]
    init_dict_flat["const_f"] = init_dict["PARAMETERS"]["const_f"]

    return init_dict_flat
