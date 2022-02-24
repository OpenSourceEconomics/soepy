import numpy as np

M_FACTOR = 4.3
Y_FACTOR = 12 * M_FACTOR


def process_tax_system(model_dict):
    # Determine taxation type:
    if "tax_splitting" not in model_dict["TAXES_TRANSFERS"].keys():
        raise ValueError("Specify if couples share taxes.")

    if "tax_year" not in model_dict["TAXES_TRANSFERS"].keys():
        raise ValueError("Specify tax_year.")

    if model_dict["TAXES_TRANSFERS"]["tax_year"] == 2007:
        tax_params = create_tax_parameters()
        model_dict["TAXES_TRANSFERS"]["tax_params"] = tax_params

    else:
        raise ValueError("Tax year not implemented.")
    return model_dict


def create_tax_parameters():
    """This function creates an array containing all paramters for the tax function."""
    thresholds = np.array([7_664, 12_739, 52_151, 250_000]) / Y_FACTOR
    rates_linear = np.array([0.15, 0.2397, 0.42, 0.45])
    rates_quadratic = np.zeros(rates_linear.shape)
    for i in range(2):
        rates_quadratic[i] = create_progressionsfactor(rates_linear, thresholds, i)
    intercepts = create_intercepts(rates_linear, rates_quadratic, thresholds)
    tax_params = np.empty((4, 4), dtype=float)
    tax_params[0, :] = thresholds
    tax_params[1, :] = intercepts
    tax_params[2, :] = rates_linear
    tax_params[3, :] = rates_quadratic
    return np.column_stack(([0, 0, 0, 0], tax_params))


def create_progressionsfactor(rates_linear, thresholds, interval_num):
    """Calculate progressionsfactor, which is the quadratic rate."""
    return (
        (rates_linear[interval_num + 1] - rates_linear[interval_num])
        / (thresholds[interval_num + 1] - thresholds[interval_num])
        / 2
    )


def create_intercepts(rates_linear, rates_quadratic, thresholds):
    # Calculate intercepts for easier caluclation
    interecepts = np.zeros(rates_linear.shape)
    for i in range(1, 4):
        interecepts[i] = (
            interecepts[i - 1]
            + (thresholds[i] - thresholds[i - 1]) * rates_linear[i - 1]
            + (thresholds[i] - thresholds[i - 1]) ** 2 * rates_quadratic[i - 1]
        )
    return interecepts


def create_child_care_costs(model_dict):
    """
    We define the child care costs as array with age bins in rows and pt/ft in
    columns. They are indexed corresponding to row/col_index = bin_num/choice_num -
    1. They only depend on the working status of the woman.
    """
    if (
        "child_care_costs" not in model_dict["TAXES_TRANSFERS"].keys()
        or "under_3" not in model_dict["TAXES_TRANSFERS"]["child_care_costs"].keys()
        or "3_to_6" not in model_dict["TAXES_TRANSFERS"]["child_care_costs"].keys()
    ):
        raise ValueError("Child care costs not properly specified.")
    else:
        child_care_costs = np.zeros((3, 2), dtype=float)
        child_care_costs[1, :] = model_dict["TAXES_TRANSFERS"]["child_care_costs"][
            "under_3"
        ]
        child_care_costs[2, :] = model_dict["TAXES_TRANSFERS"]["child_care_costs"][
            "3_to_6"
        ]
        model_dict["TAXES_TRANSFERS"]["child_care_costs"] = child_care_costs / M_FACTOR
    return model_dict


def process_ssc(model_dict):
    """This function processes the social security contributions."""
    model_dict["TAXES_TRANSFERS"]["ssc_deductions"] = np.array(
        [
            model_dict["TAXES_TRANSFERS"]["ssc_rate"],
            model_dict["TAXES_TRANSFERS"]["ssc_cap"] / Y_FACTOR,
        ]
    )
    return model_dict


def process_elterngeld(model_dict):
    """THis function scales the elterngeld from montly to weekly values."""
    model_dict["TAXES_TRANSFERS"]["elterngeld_min"] /= M_FACTOR
    model_dict["TAXES_TRANSFERS"]["elterngeld_max"] /= M_FACTOR
    return model_dict
