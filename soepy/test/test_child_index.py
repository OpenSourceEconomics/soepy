import pickle

import numpy as np
import pytest

from soepy.exogenous_processes.children import define_child_age_update_rule
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.non_consumption_utility import calculate_non_consumption_utility
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import get_integration_draws_and_weights
from soepy.shared.wages import calculate_log_wage
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.covariates import construct_covariates
from soepy.solve.create_state_space import create_child_indexes
from soepy.solve.create_state_space import pyth_create_state_space
from soepy.solve.emaxs import do_weighting_emax_scalar
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

    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)

    states, indexer = pyth_create_state_space(model_spec)

    # Create objects that depend only on the state space
    covariates = construct_covariates(states, model_spec)

    child_age_update_rule = define_child_age_update_rule(model_spec, states)

    child_state_indexes = create_child_indexes(
        states, indexer, model_spec, child_age_update_rule
    )

    draws_emax, draw_weights_emax = get_integration_draws_and_weights(model_spec)
    draws_emax = model_params.shock_sd
    # Set draws to zero to isolate the effect of child indexing
    # Solve the model in a backward induction procedure
    # Error term for continuation values is integrated out
    # numerically in a Monte Carlo procedure
    emaxs, _ = pyth_backward_induction(
        model_spec=model_spec,
        tax_splitting=model_spec.tax_splitting,
        model_params=model_params,
        states=states,
        child_state_indexes=child_state_indexes,
        draws=draws_emax,
        draw_weights=draw_weights_emax,
        covariates=covariates,
        prob_child=prob_child,
        prob_partner=prob_partner,
        is_expected=True,
    )

    return states, emaxs, child_state_indexes, prob_child, prob_partner


def test_child_state_index(input_data):
    states, emaxs, child_state_indexes, prob_child, prob_partner = input_data

    (
        period,
        educ_level,
        choice_lagged,
        exp_p,
        exp_f,
        disutil_type,
        age_kid,
        partner_indicator,
    ) = states[10]

    child_index = child_state_indexes[10, 1, :, :]
    child_emax = emaxs[:, 3][child_index]

    weighted_emax = do_weighting_emax_scalar(
        child_emax,
        prob_child[period, educ_level],
        prob_partner[period, educ_level, partner_indicator, :],
    )

    np.testing.assert_allclose(weighted_emax, emaxs[10, 1])
