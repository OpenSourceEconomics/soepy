import collections

import yaml


def read_init_file(init_file_name):
    """Reads in the model specification from yaml file"""

    # Import yaml initialization file as dictionary init_dict
    if isinstance(init_file_name, str):
        with open(init_file_name) as y:
            init_dict = yaml.load(y, Loader=yaml.FullLoader)
    else:
        init_dict = init_file_name

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
    shocks_cov = [
        init_dict["PARAMETERS"]["sigma_1"],
        init_dict["PARAMETERS"]["sigma_2"],
        init_dict["PARAMETERS"]["sigma_3"],
    ]
    shocks_cov = [shocks_cov[0] ** 2, shocks_cov[1] ** 2, shocks_cov[2] ** 2]

    # Extract the number of types

    if [v for k, v in init_dict["PARAMETERS"].items() if "share" in k] == 1.0:
        num_types = 1
        type_shares = 1.0
    else:
        num_types = (
            len([v for k, v in init_dict["PARAMETERS"].items() if "share" in k]) + 1
        )

        # Aggregate type shares in list object
        type_shares = [
            1 - sum([v for k, v in init_dict["PARAMETERS"].items() if "share" in k])
        ] + [v for k, v in init_dict["PARAMETERS"].items() if "share" in k]

    # Append derived attributes to init_dict
    init_dict["DERIVED_ATTR"] = {
        "educ_range": educ_range,
        "shocks_cov": shocks_cov,
        "num_types": num_types,
        "type_shares": type_shares,
    }

    # Return function output
    return init_dict


def create_namedtuple(init_dict):
    """Transfers model specification from a dictionary
    to a named tuple class object"""

    init_dict_flat = flatten_init_dict(init_dict)

    init_dict_flat = group_parameters(init_dict, init_dict_flat)

    model_params = dict_to_namedtuple(init_dict_flat)

    return model_params


def flatten_init_dict(init_dict):
    """Removes the grouping from the nested dictionary"""

    groups = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "DERIVED_ATTR",
    ]

    init_dict_flat = dict()

    for group in groups:

        keys_ = list(init_dict[group].keys())
        values_ = list(init_dict[group].values())

        for k_, key_ in enumerate(keys_):

            init_dict_flat[key_] = values_[k_]

    return init_dict_flat


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

    init_dict_flat["theta_p"] = [
        v for k, v in init_dict["PARAMETERS"].items() if "theta_p" in k
    ]
    init_dict_flat["theta_f"] = [
        v for k, v in init_dict["PARAMETERS"].items() if "theta_f" in k
    ]

    for i in range(1, init_dict["DERIVED_ATTR"]["num_types"]):
        init_dict_flat["share_" + str(i)] = init_dict["PARAMETERS"]["share_" + str(i)]

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

    return init_dict_flat


def dict_to_namedtuple(dictionary):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple("model_parameters", dictionary.keys())(**dictionary)
