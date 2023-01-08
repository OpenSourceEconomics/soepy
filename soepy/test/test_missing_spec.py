import copy
import pickle

import pytest

from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.soepy_config import TEST_RESOURCES_DIR


@pytest.fixture(scope="module")
def input_data():
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)
    (
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
        expected_df,
        expected_df_unbiased,
    ) = tests[0]

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
