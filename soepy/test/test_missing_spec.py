import copy

import pandas as pd
import pytest
import yaml

from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.test.random_init import random_init


@pytest.fixture(scope="module")
def input_data():
    """Provide a valid baseline spec/params using `random_init`."""

    random_init(
        {
            "AGENTS": 10,
            "PERIODS": 3,
            "CHILD_AGE_INIT_MAX": 1,
            "INIT_EXP_MAX": 1,
            "SEED_SIM": 1111,
            "SEED_EMAX": 2222,
            "NUM_DRAWS_EMAX": 5,
        }
    )

    with open("test.soepy.yml") as f:
        model_spec_init_dict = yaml.load(f, Loader=yaml.Loader)

    random_model_params_df = pd.read_pickle("test.soepy.pkl")
    return model_spec_init_dict, random_model_params_df


def test_missing_tax_year(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    del local_init_dict["TAXES_TRANSFERS"]["tax_year"]
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Specify tax_year."


def test_missing_child_wrong_tax_year(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    local_init_dict["TAXES_TRANSFERS"]["tax_year"] = 2006
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Tax year not implemented."


def test_missing_child_care_costs(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    del local_init_dict["TAXES_TRANSFERS"]["child_care_costs"]
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Child care costs not properly specified."


def test_missing_child_care_costs_under_3(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    del local_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["under_3"]
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Child care costs not properly specified."


def test_missing_child_care_costs_under_3_6(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    del local_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["3_to_6"]
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Child care costs not properly specified."


def test_missing_tax_splitting(input_data):
    model_spec_init_dict, random_model_params_df = input_data
    local_init_dict = copy.deepcopy(model_spec_init_dict)
    del local_init_dict["TAXES_TRANSFERS"]["tax_splitting"]
    with pytest.raises(ValueError) as error_info:
        read_model_spec_init(local_init_dict, random_model_params_df)
    assert str(error_info.value) == "Specify if couples share taxes."
