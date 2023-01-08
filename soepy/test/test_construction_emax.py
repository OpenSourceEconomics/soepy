import pickle

import numpy as np
import pytest

from soepy.exogenous_processes.children import define_child_age_update_rule
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.shared_auxiliary import calculate_log_wage
from soepy.shared.shared_auxiliary import calculate_non_consumption_utility
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
        expected_df,
    ) = tests[0]
    model_spec_init_dict["SOLUTION"]["num_draws_emax"] = 1

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    # Type 1 never wants to work!
    random_model_params_df.loc[("hetrg_unobs", "theta_p1"), "value"] *= -50
    random_model_params_df.loc[("hetrg_unobs", "theta_f1"), "value"] *= -50

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

    draws_emax *= 0

    log_wage_systematic = calculate_log_wage(model_params, states, True)

    non_consumption_utilities = calculate_non_consumption_utility(
        model_params.theta_p,
        model_params.theta_f,
        model_params.no_kids_f,
        model_params.no_kids_p,
        model_params.yes_kids_f,
        model_params.yes_kids_p,
        model_params.child_0_2_f,
        model_params.child_0_2_p,
        model_params.child_3_5_f,
        model_params.child_3_5_p,
        model_params.child_6_10_f,
        model_params.child_6_10_p,
        states,
        covariates[:, 0],
        np.array([0, 1, 2], dtype=float),
    )

    deductions_spec = np.array(model_spec.ssc_deductions)
    tax_splitting = model_spec.tax_splitting

    non_employment_consumption_resources = (
        calculate_non_employment_consumption_resources(
            deductions_spec,
            model_spec.tax_params,
            model_spec,
            states,
            log_wage_systematic,
            covariates[:, 1],
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
        model_spec,
        model_params,
        emaxs,
        states,
        indexer,
        covariates,
        non_employment_consumption_resources,
        non_consumption_utilities,
    )


@pytest.fixture(scope="module")
def states_tested(input_data):
    (
        model_spec,
        model_params,
        emaxs,
        states,
        indexer,
        covariates,
        non_employment_consumption_resources,
        non_consumption_utilities,
    ) = input_data
    # Get states from type 1
    states_selected = states[(states[:, 5] == 1)]
    rand_states = np.random.randint(0, states_selected.shape[0], size=100)
    return rand_states


def test_construct_emax(input_data, states_tested):
    (
        model_spec,
        model_params,
        emaxs,
        states,
        indexer,
        covariates,
        non_employment_consumption_resources,
        non_consumption_utilities,
    ) = input_data
    # Get states from type 1
    states_selected = states[(states[:, 5] == 1)]

    for test_state in states_tested:

        (
            period,
            educ_level,
            lagged_choice,
            exp_pt,
            exp_ft,
            type_1,
            age_young_child,
            partner_ind,
        ) = states_selected[test_state, :]
        assert type_1 == 1

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
        equ_scale = covariates[ind_state, 2]
        non_employ_cons = non_employment_consumption_resources[ind_state] / equ_scale
        mu = model_params.mu
        consumption_utility = non_employ_cons**mu / mu
        value_func = consumption_utility + model_params.delta * emaxs[ind_state, 0]
        np.testing.assert_equal(value_func, emaxs[ind_state, 3])
