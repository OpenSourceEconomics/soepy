import pickle

import numpy as np
import pytest

from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_log_wage
from soepy.shared.shared_auxiliary import calculate_net_income
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.create_state_space import create_state_space_objects


@pytest.fixture(scope="module")
def input_data():
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

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    log_wage_systematic = calculate_log_wage(model_params, states, False)

    tax_splitting = model_spec.tax_splitting

    non_employment_consumption_resources = (
        calculate_non_employment_consumption_resources(
            model_spec.ssc_deductions,
            model_spec.tax_params,
            model_spec,
            states,
            log_wage_systematic,
            covariates[:, 1],
            covariates[:, 3],
            tax_splitting,
        )
    )
    return (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    )


def test_male_wages(input_data):
    """This tests that in all married states there are male wages"""
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    np.testing.assert_array_equal(states[:, 7] == 1, covariates[:, 1] > 0)


def test_child_present(input_data):
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    np.testing.assert_equal(covariates[:, 0] != 0, states[:, 6] > -1)


def test_child_update_rule_no_child(input_data):
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    no_child = states[:, 6] == -1
    np.testing.assert_array_equal(
        states[no_child][:, 6], child_age_update_rule[no_child]
    )


def test_child_update_rule_aging_child(input_data):
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    aging_child = (states[:, 6] > -1) & (states[:, 6] <= model_spec.child_age_max)
    np.testing.assert_array_equal(
        states[aging_child][:, 6] + 1, child_age_update_rule[aging_child]
    )


def test_non_consumption_resources_married_no_newborn(input_data):
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    married = states[:, 7] == 1
    working_ft_last_period = states[:, 2] == 2
    working_pt_last_period = states[:, 2] == 1
    working_last_period = working_ft_last_period | working_pt_last_period
    newborn_child = states[:, 6] == 0
    subgroup_check = married & ~working_last_period & ~newborn_child
    married_non_emplyed = non_employment_consumption_resources[subgroup_check]
    relevant_male_wages = covariates[:, 1][subgroup_check]

    for i in range(married_non_emplyed.shape[0]):
        assert (
            calculate_net_income(
                np.array(model_spec.tax_params),
                np.array(model_spec.ssc_deductions),
                0,
                relevant_male_wages[i],
                model_spec.tax_splitting,
            )
            == married_non_emplyed[i]
        )


def test_work_choices(input_data):
    (
        covariates,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    working_ft_last_period = states[:, 2] == 2
    working_pt_last_period = states[:, 2] == 1
    working_last_period = working_ft_last_period | working_pt_last_period
    np.testing.assert_array_equal(~working_last_period, states[:, 2] == 0)
