import itertools
import pickle
import random

import numpy as np
import pytest

from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.shared_auxiliary import calculate_log_wage
from soepy.shared.shared_auxiliary import calculate_non_consumption_utility
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.create_state_space import create_state_space_objects


CASES_TEST = random.sample(range(0, 100), 10)
SUBJ_BELIEFS = [True, False]


@pytest.fixture(scope="module")
def input_vault():
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    return tests


@pytest.mark.parametrize(
    "test_id, is_expected", itertools.product(CASES_TEST, SUBJ_BELIEFS)
)
def test_pyth_simulate(input_vault, test_id, is_expected):
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
    ) = input_vault[test_id]

    model_params_df, model_params = read_model_params_init(random_model_params_df)
    model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    # Calculate utility components
    log_wage_systematic = calculate_log_wage(model_params, states, is_expected)

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
        covariates,
        np.array([0, 1, 2], dtype=float),
    )

    for edu_ind, edu_type in enumerate(["low", "middle", "high"]):
        # Now test the full and part time level, 1 and 2 for the wages
        relevant_states_ind = (
            (states[:, 3] == 1) & (states[:, 4] == 2) & (states[:, 1] == edu_ind)
        )

        gamma_0 = random_model_params_df.loc[
            ("const_wage_eq", f"gamma_0_{edu_type}"), "value"
        ]
        gamma_f = random_model_params_df.loc[
            ("exp_returns_f", f"gamma_f_{edu_type}"), "value"
        ]
        if is_expected:
            gamma_p = (
                random_model_params_df.loc[
                    ("exp_returns_p_bias", f"gamma_p_bias_{edu_type}"), "value"
                ]
                * gamma_f
            )
        else:
            gamma_p = random_model_params_df.loc[
                ("exp_returns_p", f"gamma_p_{edu_type}"), "value"
            ]

        wage_calc = gamma_0 + gamma_f * np.log(3) + gamma_p * np.log(2)

        np.testing.assert_array_equal(
            log_wage_systematic[relevant_states_ind],
            np.full(log_wage_systematic[relevant_states_ind].shape, wage_calc),
        )
