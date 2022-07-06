import pickle

import numpy as np
import pytest

from soepy.exogenous_processes.children import define_child_age_update_rule
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.non_employment_benefits import calculate_non_employment_benefits
from soepy.shared.shared_auxiliary import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_utility_components
from soepy.shared.shared_auxiliary import draw_disturbances
from soepy.shared.shared_constants import HOURS
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.covariates import construct_covariates
from soepy.solve.create_state_space import create_child_indexes
from soepy.solve.create_state_space import pyth_create_state_space
from soepy.solve.solve_python import get_integration_draws_and_weights
from soepy.solve.solve_python import pyth_backward_induction


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
        expected_df_sim_func,
        expected_df_sim_sol,
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

    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)

    states, indexer = pyth_create_state_space(model_spec)

    # Create objects that depend only on the state space
    covariates = construct_covariates(states, model_spec)

    child_age_update_rule = define_child_age_update_rule(model_spec, states)

    child_state_indexes = create_child_indexes(
        states, indexer, model_spec, child_age_update_rule
    )

    draws_emax, draw_weights_emax = get_integration_draws_and_weights(
        model_spec, model_params
    )

    log_wage_systematic, non_consumption_utilities = calculate_utility_components(
        model_params, model_spec, states, covariates, True
    )

    non_employment_benefits = calculate_non_employment_benefits(
        model_spec, states, log_wage_systematic
    )

    deductions_spec = np.array(model_spec.ssc_deductions)
    tax_splitting = model_spec.tax_splitting

    non_employment_consumption_resources = (
        calculate_non_employment_consumption_resources(
            deductions_spec,
            model_spec.tax_params,
            covariates[:, 1],
            non_employment_benefits,
            tax_splitting,
        )
    )

    index_child_care_costs = np.where(covariates[:, 0] > 2, 0, covariates[:, 0]).astype(
        int
    )

    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs = pyth_backward_induction(
        model_spec.num_periods,
        tax_splitting,
        model_params.mu,
        model_params.delta,
        model_spec.tax_params,
        states,
        HOURS,
        model_spec.child_care_costs,
        child_state_indexes,
        log_wage_systematic,
        non_consumption_utilities,
        draws_emax,
        draw_weights_emax,
        covariates,
        index_child_care_costs,
        prob_child,
        prob_partner,
        non_employment_consumption_resources,
        model_spec.ssc_deductions,
    )

    return (
        emaxs,
        states,
        indexer,
        prob_child,
        prob_partner,
        child_age_update_rule,
    )


def test_emaxs_married(input_data):
    (
        emaxs,
        states,
        indexer,
        prob_child,
        prob_partner,
        child_age_update_rule,
    ) = input_data
    # Get states from period 1, type 1, married and no kid
    states_selected = states[
        (states[:, 0] == 1) & (states[:, 6] == -1) & (states[:, 7] == 1)
    ]
    rand_state = np.random.randint(0, states_selected.shape[0])
    (
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ) = states_selected[rand_state, :]
    assert period == 1
    assert age_young_child == -1
    assert partner_ind == 1
    ind_state = indexer[
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ]
    emax_cstate_sep_no_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, -1, 0], 3
    ]
    emax_cstate_sep_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 0], 3
    ]
    emax_cstate_mar_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 1], 3
    ]
    emax_cstate_mar_no_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, -1, 1], 3
    ]
    p_child_arr = prob_child[period, educ_level]
    p_part_sep = prob_partner[period, educ_level, 1, 0]

    w_child_single = p_child_arr * p_part_sep * emax_cstate_sep_kid
    w_no_child_single = (1 - p_child_arr) * p_part_sep * emax_cstate_sep_no_kid
    w_child_married = p_child_arr * (1 - p_part_sep) * emax_cstate_mar_kid
    w_no_child_married = (1 - p_child_arr) * (1 - p_part_sep) * emax_cstate_mar_no_kid

    weighted_sum = (
        w_child_single + w_no_child_single + w_child_married + w_no_child_married
    )
    np.testing.assert_almost_equal(weighted_sum, emaxs[ind_state, 0])


def test_emaxs_single(input_data):
    (
        emaxs,
        states,
        indexer,
        prob_child,
        prob_partner,
        child_age_update_rule,
    ) = input_data
    # Get states from period 1, type 1, not married and no kid
    states_selected = states[
        (states[:, 0] == 1) & (states[:, 6] == -1) & (states[:, 7] == 0)
    ]
    rand_state = np.random.randint(0, states_selected.shape[0])
    (
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ) = states_selected[rand_state, :]
    assert period == 1
    assert age_young_child == -1
    assert partner_ind == 0
    ind_state = indexer[
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ]
    emax_cstate_sep_no_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, -1, 0], 3
    ]
    emax_cstate_sep_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 0], 3
    ]
    emax_cstate_mar_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 1], 3
    ]
    emax_cstate_mar_no_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, -1, 1], 3
    ]
    p_child_arr = prob_child[period, educ_level]
    p_part_arriv = prob_partner[period, educ_level, 0, 1]

    w_child_single = p_child_arr * (1 - p_part_arriv) * emax_cstate_sep_kid
    w_no_child_single = (1 - p_child_arr) * (1 - p_part_arriv) * emax_cstate_sep_no_kid
    w_child_married = p_child_arr * p_part_arriv * emax_cstate_mar_kid
    w_no_child_married = (1 - p_child_arr) * p_part_arriv * emax_cstate_mar_no_kid

    weighted_sum = (
        w_child_single + w_no_child_single + w_child_married + w_no_child_married
    )
    np.testing.assert_almost_equal(weighted_sum, emaxs[ind_state, 0])


def test_emaxs_single_with_kid(input_data):
    (
        emaxs,
        states,
        indexer,
        prob_child,
        prob_partner,
        child_age_update_rule,
    ) = input_data
    # Get states from period 1, type 1, married and kids
    states_selected = states[
        (states[:, 0] == 1) & (states[:, 6] != -1) & (states[:, 7] == 0)
    ]
    rand_state = np.random.randint(0, states_selected.shape[0])
    (
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ) = states_selected[rand_state, :]
    assert period == 1
    assert age_young_child != -1
    assert partner_ind == 0
    ind_state = indexer[
        period,
        educ_level,
        lagged_choice,
        exp_pt,
        exp_ft,
        type_1,
        age_young_child,
        partner_ind,
    ]

    emax_cstate_sep_no_kid_arr = emaxs[
        indexer[
            period + 1,
            educ_level,
            0,
            exp_pt,
            exp_ft,
            type_1,
            child_age_update_rule[ind_state],
            0,
        ],
        3,
    ]
    emax_cstate_mar_no_kid_arr = emaxs[
        indexer[
            period + 1,
            educ_level,
            0,
            exp_pt,
            exp_ft,
            type_1,
            child_age_update_rule[ind_state],
            1,
        ],
        3,
    ]

    emax_cstate_sep_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 0], 3
    ]
    emax_cstate_mar_kid = emaxs[
        indexer[period + 1, educ_level, 0, exp_pt, exp_ft, type_1, 0, 1], 3
    ]

    p_child_arr = prob_child[period, educ_level]
    p_part_arriv = prob_partner[period, educ_level, 0, 1]

    w_child_single = p_child_arr * (1 - p_part_arriv) * emax_cstate_sep_kid
    w_no_child_single = (
        (1 - p_child_arr) * (1 - p_part_arriv) * emax_cstate_sep_no_kid_arr
    )
    w_child_married = p_child_arr * p_part_arriv * emax_cstate_mar_kid
    w_no_child_married = (1 - p_child_arr) * p_part_arriv * emax_cstate_mar_no_kid_arr

    weighted_sum = (
        w_child_single + w_no_child_single + w_child_married + w_no_child_married
    )
    np.testing.assert_almost_equal(weighted_sum, emaxs[ind_state, 0])
