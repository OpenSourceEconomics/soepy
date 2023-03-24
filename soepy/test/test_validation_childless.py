"""This test looks at single women only."""
import pickle

import numpy as np
import pandas as pd
import pytest

from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve


@pytest.fixture(scope="module")
def input_data():
    out = {}

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
    ) = tests[6]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    model_params_df, model_params = read_model_params_init(random_model_params_df)
    ccc_under_3 = np.array(
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["under_3"]
    )
    ccc_3_to_6 = np.array(
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["3_to_6"]
    )

    for name, regime in [("original", "elterngeld"), ("validation", "erziehungsgeld")]:

        model_spec_init_dict["TAXES_TRANSFERS"]["parental_leave_regime"] = regime

        model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)

        prob_educ_level = gen_prob_educ_level_vector(model_spec)
        prob_child_age = []
        prob_child_age_old = gen_prob_child_init_age_vector(model_spec)
        for educ_level_list in prob_child_age_old:
            educ_level_array = np.array(educ_level_list)
            educ_level_array[0] = 1
            educ_level_array[1:] = 0
            prob_child_age += [educ_level_array]
        prob_partner_present = gen_prob_partner_present_vector(model_spec)
        prob_exp_ft = gen_prob_init_exp_vector(
            model_spec, model_spec.ft_exp_shares_file_name
        )
        prob_exp_pt = gen_prob_init_exp_vector(
            model_spec, model_spec.pt_exp_shares_file_name
        )
        prob_child = gen_prob_child_vector(model_spec)
        prob_child[:, :] = 0
        prob_partner = gen_prob_partner(model_spec)
        prob_partner[:, 0, 1] = 0
        prob_partner[:, 0, 0] = 1
        prob_partner_present[:] = 0

        # Create state space
        (
            states,
            indexer,
            covariates,
            child_age_update_rule,
            child_state_indexes,
        ) = create_state_space_objects(model_spec)

        # Obtain model solution
        (non_consumption_utilities, emaxs,) = pyth_solve(
            states,
            covariates,
            child_state_indexes,
            model_params,
            model_spec,
            prob_child,
            prob_partner,
            False,
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
            prob_exp_ft,
            prob_exp_pt,
            prob_child,
            prob_partner,
            is_expected=False,
        )

        out[name] = calculated_df
        out[f"{name}_emax"] = emaxs
        out[f"{name}_covs"] = covariates
    return out


def test_childless(input_data):
    pd.testing.assert_frame_equal(input_data["original"], input_data["validation"])


def test_childless_emax(input_data):
    not_having_kids = input_data["original_covs"][:, 3] == 0
    np.testing.assert_almost_equal(
        input_data["original_emax"][not_having_kids, :],
        input_data["validation_emax"][not_having_kids, :],
    )
