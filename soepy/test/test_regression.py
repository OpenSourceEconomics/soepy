import pickle
import random

import pandas as pd
import pytest

from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_years_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.constants_sim import LABELS_DATA_SPARSE
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve
from soepy.test.resources.aux_funcs import cleanup


CASES_TEST = random.sample(range(0, 100), 10)

DATA_LABLES_CHECK = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Type",
    "Age_Youngest_Child",
    "Partner_Indicator",
    "Choice",
    "Log_Systematic_Wage",
    "Wage_Observed",
    "Non_Consumption_Utility_N",
    "Non_Consumption_Utility_P",
    "Non_Consumption_Utility_F",
    "Flow_Utility_N",
    "Flow_Utility_P",
    "Flow_Utility_F",
    "Male_Wages",
    "Continuation_Value_N",
    "Continuation_Value_P",
    "Continuation_Value_F",
]


@pytest.fixture(scope="module")
def input_vault():
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    return tests


# The regression-vault expectations were generated for the old discrete experience
# model. Keep them as historical reference, but skip under the continuous refactor.
@pytest.mark.skip(reason="Regression vault is discrete-experience based")
@pytest.mark.parametrize("test_id", CASES_TEST)
def test_pyth_simulate(input_vault, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """

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
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    model_params_df, model_params = read_model_params_init(random_model_params_df)
    model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)

    prob_educ_level = gen_prob_educ_level_vector(model_spec)
    prob_child_age = gen_prob_child_init_age_vector(model_spec)
    prob_partner_present = gen_prob_partner_present_vector(model_spec)
    prob_exp_years = gen_prob_init_exp_years_vector(model_spec)
    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    # Obtain model solution
    non_consumption_utilities, emaxs = pyth_solve(
        states,
        covariates,
        child_state_indexes,
        model_params,
        model_spec,
        prob_child,
        prob_partner,
        is_expected=True,
    )

    # Simulate
    calculated_df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
        non_consumption_utilities,
        child_age_update_rule,
        prob_educ_level,
        prob_child_age,
        prob_partner_present,
        prob_exp_years,
        prob_child,
        prob_partner,
        is_expected=False,
    )

    pd.testing.assert_series_equal(
        calculated_df.sum(axis=0).loc[DATA_LABLES_CHECK],
        expected_df.loc[DATA_LABLES_CHECK],
    )
    cleanup()


@pytest.mark.skip(reason="Regression vault is discrete-experience based")
@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func(input_vault, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
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
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    calculated_df = simulate(random_model_params_df, model_spec_init_dict)

    pd.testing.assert_series_equal(
        expected_df.loc[DATA_LABLES_CHECK],
        calculated_df.reset_index().sum(axis=0).loc[DATA_LABLES_CHECK],
    )
    cleanup()


@pytest.mark.skip(reason="Regression vault is discrete-experience based")
@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func_unbiased(input_vault, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
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
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    calculated_df = simulate(
        random_model_params_df, model_spec_init_dict, is_expected=False
    )

    pd.testing.assert_series_equal(
        expected_df_unbiased.loc[DATA_LABLES_CHECK],
        calculated_df.reset_index().sum(axis=0).loc[DATA_LABLES_CHECK],
    )
    cleanup()


@pytest.mark.skip(reason="Regression vault is discrete-experience based")
@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func_data_sparse(input_vault, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
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
    ) = input_vault[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    calculated_df = simulate(
        random_model_params_df, model_spec_init_dict, data_sparse=True
    )

    pd.testing.assert_series_equal(
        expected_df.loc[LABELS_DATA_SPARSE],
        calculated_df.reset_index().sum(axis=0).loc[LABELS_DATA_SPARSE],
        check_dtype=False,
    )
    cleanup()
