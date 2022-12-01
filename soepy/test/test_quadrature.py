"""This test looks at single women only."""
import pickle

import numpy as np
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
from soepy.test.resources.aux_funcs import create_disc_sum_av_utility


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

    for name, monte in [("monte-carlo", True), ("quadrature", False)]:
        if monte:
            pass
        else:
            model_spec_init_dict["SOLUTION"]["integration_method"] = "quadrature"
            model_spec_init_dict["SOLUTION"]["num_draws_emax"] = 21

        model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)

        prob_educ_level = gen_prob_educ_level_vector(model_spec)
        prob_child_age = gen_prob_child_init_age_vector(model_spec)
        prob_partner_present = gen_prob_partner_present_vector(model_spec)
        prob_exp_ft = gen_prob_init_exp_vector(
            model_spec, model_spec.ft_exp_shares_file_name
        )
        prob_exp_pt = gen_prob_init_exp_vector(
            model_spec, model_spec.pt_exp_shares_file_name
        )
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
        (
            non_employment_consumption_resources,
            non_consumption_utilities,
            emaxs,
        ) = pyth_solve(
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
            non_employment_consumption_resources,
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

        out[name] = create_disc_sum_av_utility(
            calculated_df, model_params_df.loc[("discount", "delta"), "value"]
        )

    return out


def test_single_woman(input_data):
    np.testing.assert_equal(input_data["quadrature"], input_data["monte-carlo"])
