import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_log_wage
from soepy.shared.shared_constants import HOURS
from soepy.shared.tax_and_transfers import calculate_net_income
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
        log_wage_systematic,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    )


def test_married_no_newborn(input_data):
    (
        covariates,
        log_wage_systematic,
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
    transfers_check = non_employment_consumption_resources[subgroup_check]
    relevant_male_wages = covariates[:, 1][subgroup_check]

    for i in range(transfers_check.shape[0]):
        male_net_wage = calculate_net_income(
            np.array(model_spec.tax_params),
            np.array(model_spec.ssc_deductions),
            0,
            relevant_male_wages[i],
            model_spec.tax_splitting,
        )
        assert_allclose(
            male_net_wage,
            transfers_check[i],
        )


def test_not_married_no_newborn_allerziehend(input_data):
    (
        covariates,
        log_wage_systematic,
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
    child = states[:, 6] > -1
    subgroup_check = ~married & ~working_last_period & ~newborn_child & child

    transfers_check = non_employment_consumption_resources[subgroup_check]

    alg_2_alleinerziehend = (
        model_spec.regelsatz_single
        + model_spec.regelsatz_child
        + model_spec.addition_child_single
        + model_spec.housing_single
        + model_spec.housing_addtion
    )

    assert_allclose(
        alg_2_alleinerziehend,
        transfers_check,
    )


def test_alg2_single(input_data):
    (
        covariates,
        log_wage_systematic,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data
    married = states[:, 7] == 1
    working_ft_last_period = states[:, 2] == 2
    working_pt_last_period = states[:, 2] == 1
    working_last_period = working_ft_last_period | working_pt_last_period
    child = states[:, 6] > -1
    subgroup_check = ~married & ~working_last_period & ~child

    transfers_check = non_employment_consumption_resources[subgroup_check]

    alg2_single = model_spec.regelsatz_single + model_spec.housing_single

    assert_allclose(
        alg2_single,
        transfers_check,
    )


@pytest.mark.parametrize("work_choice", [1, 2])
def test_alg1_no_child(input_data, work_choice):
    (
        covariates,
        log_wage_systematic,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data

    working_ft_last_period = states[:, 2] == work_choice

    no_child = states[:, 6] == -1
    subgroup_check = working_ft_last_period & no_child

    transfers_check = non_employment_consumption_resources[subgroup_check]

    prox_net_wage_systematic = (0.65 * np.exp(log_wage_systematic))[subgroup_check]
    alg2_single = model_spec.regelsatz_single + model_spec.housing_single

    relevant_male_wages = covariates[:, 1][subgroup_check]

    for i in range(transfers_check.shape[0]):
        male_net_wage = calculate_net_income(
            np.array(model_spec.tax_params),
            np.array(model_spec.ssc_deductions),
            0,
            relevant_male_wages[i],
            model_spec.tax_splitting,
        )
        assert_allclose(
            max(
                male_net_wage
                + model_spec.alg1_replacement_no_child
                * prox_net_wage_systematic[i]
                * HOURS[work_choice],
                alg2_single,
            ),
            transfers_check[i],
        )


@pytest.mark.parametrize("work_choice", [1, 2])
def test_alg1_child(input_data, work_choice):
    (
        covariates,
        log_wage_systematic,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data

    working_ft_last_period = states[:, 2] == work_choice

    child = states[:, 6] > -1
    new_born_child = states[:, 6] == 0
    subgroup_check = working_ft_last_period & child & ~new_born_child

    transfers_check = non_employment_consumption_resources[subgroup_check]

    prox_net_wage_systematic = (0.65 * np.exp(log_wage_systematic))[subgroup_check]

    relevant_male_wages = covariates[:, 1][subgroup_check]
    alg_2_alleinerziehend = (
        model_spec.regelsatz_single
        + model_spec.regelsatz_child
        + model_spec.addition_child_single
        + model_spec.housing_single
        + model_spec.housing_addtion
    )

    for i in range(transfers_check.shape[0]):
        male_net_wage = calculate_net_income(
            np.array(model_spec.tax_params),
            np.array(model_spec.ssc_deductions),
            0,
            relevant_male_wages[i],
            model_spec.tax_splitting,
        )
        assert_allclose(
            max(
                male_net_wage
                + model_spec.alg1_replacement_child
                * prox_net_wage_systematic[i]
                * HOURS[work_choice]
                + model_spec.child_benefits,
                alg_2_alleinerziehend,
            ),
            transfers_check[i],
        )


@pytest.mark.parametrize("work_choice", [1, 2])
def test_elterngeld(input_data, work_choice):
    (
        covariates,
        log_wage_systematic,
        states,
        non_employment_consumption_resources,
        model_spec,
        child_age_update_rule,
    ) = input_data

    working_choice_last_period = states[:, 2] == work_choice

    new_born_child = states[:, 6] == 0
    subgroup_check = working_choice_last_period & new_born_child

    transfers_check = non_employment_consumption_resources[subgroup_check]

    prox_net_wage_systematic = (0.65 * np.exp(log_wage_systematic))[subgroup_check]
    relevant_male_wages = covariates[:, 1][subgroup_check]
    alg_2_alleinerziehend = (
        model_spec.regelsatz_single
        + model_spec.regelsatz_child
        + model_spec.addition_child_single
        + model_spec.housing_single
        + model_spec.housing_addtion
    )

    for i in range(transfers_check.shape[0]):

        elterngeld_without_child_benefits = np.minimum(
            np.maximum(
                model_spec.elterngeld_replacement
                * prox_net_wage_systematic[i]
                * HOURS[work_choice],
                model_spec.elterngeld_min,
            ),
            model_spec.elterngeld_max,
        )
        male_net_wage = calculate_net_income(
            np.array(model_spec.tax_params),
            np.array(model_spec.ssc_deductions),
            0,
            relevant_male_wages[i],
            model_spec.tax_splitting,
        )
        assert_allclose(
            max(
                male_net_wage
                + elterngeld_without_child_benefits
                + model_spec.child_benefits,
                alg_2_alleinerziehend,
            ),
            transfers_check[i],
        )
