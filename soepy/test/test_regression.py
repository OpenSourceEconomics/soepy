import pickle
import random

import pandas as pd

from soepy.soepy_config import TEST_RESOURCES_DIR
from development.tests.auxiliary.auxiliary import cleanup

from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner_arrival
from soepy.exogenous_processes.partner import gen_prob_partner_separation
from soepy.simulate.simulate_python import simulate
from soepy.solve.solve_python import pyth_solve
from soepy.simulate.simulate_auxiliary import pyth_simulate
import pytest
from soepy.shared.shared_constants import DATA_LABLES_SIM
import itertools


CASES_TEST = random.sample(range(0, 100), 10)


@pytest.fixture(scope="module")
def input_data():
    vault = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    with open(vault, "rb") as file:
        tests = pickle.load(file)

    solve_dict = {}
    df_dict = {}

    for i in CASES_TEST:
        df_dict[i] = {}

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
            expected_df_sim_sol,
            df_dict[i]["expected_df"],
        ) = tests[i]

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
        prob_exp_ft = gen_prob_init_exp_vector(
            model_spec, model_spec.ft_exp_shares_file_name
        )
        prob_exp_pt = gen_prob_init_exp_vector(
            model_spec, model_spec.pt_exp_shares_file_name
        )
        prob_child = gen_prob_child_vector(model_spec)
        prob_partner_arrival = gen_prob_partner_arrival(model_spec)
        prob_partner_separation = gen_prob_partner_separation(model_spec)

        solve_dict[i] = {}
        # Solve
        (
            solve_dict[i]["states"],
            solve_dict[i]["indexer"],
            solve_dict[i]["covariates"],
            solve_dict[i]["non_employment_consumption_resources"],
            solve_dict[i]["emaxs"],
            solve_dict[i]["child_age_update_rule"],
            solve_dict[i]["deductions_spec"],
            solve_dict[i]["income_tax_spec"],
        ) = pyth_solve(
            model_params,
            model_spec,
            prob_child,
            prob_partner_arrival,
            prob_partner_separation,
            is_expected=False,
        )

        # Simulate
        df_dict[i]["calculated_df"] = pyth_simulate(
            model_params,
            model_spec,
            solve_dict[i]["states"],
            solve_dict[i]["indexer"],
            solve_dict[i]["emaxs"],
            solve_dict[i]["covariates"],
            solve_dict[i]["non_employment_consumption_resources"],
            solve_dict[i]["deductions_spec"],
            solve_dict[i]["income_tax_spec"],
            solve_dict[i]["child_age_update_rule"],
            prob_educ_level,
            prob_child_age,
            prob_partner_present,
            prob_exp_ft,
            prob_exp_pt,
            prob_child,
            prob_partner_arrival,
            prob_partner_separation,
            is_expected=False,
        )
    return solve_dict, df_dict


@pytest.mark.parametrize(
    "test_col, test_id", itertools.product(DATA_LABLES_SIM, CASES_TEST)
)
def test_pyth_simulate(input_data, test_col, test_id):
    """This test runs a random selection of test regression tests from
    our regression test battery.
    """
    solve_dict, df_dict = input_data
    pd.testing.assert_series_equal(
        df_dict[test_id]["calculated_df"][test_col],
        df_dict[test_id]["expected_df"][test_col],
    )


@pytest.mark.parametrize("test_id", CASES_TEST)
def test_simulation_func(test_id):
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
        expected_df_sim_func,
        expected_df_sim_sol,
    ) = tests[test_id]

    exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
    exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
    exog_child_info.to_pickle("test.soepy.child.pkl")
    exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
    exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
    exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
    exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
    exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

    calculated_df = simulate(random_model_params_df, model_spec_init_dict)

    for col in expected_df_sim_func.columns.tolist():
        pd.testing.assert_series_equal(
            expected_df_sim_func[col],
            calculated_df[col],
        )
    cleanup()


# def save_new_fault():
#     sim_func_vault = TEST_RESOURCES_DIR / "regression_vault_sim_func.pkl"
#
#     unified_vault = {}
#     with open(sim_func_vault, "rb") as file:
#         tests_sim_func = pickle.load(file)
#
#     sim_func_vault = TEST_RESOURCES_DIR / "regression_vault_sim_sol.pkl"
#
#     with open(sim_func_vault, "rb") as file:
#         tests_sim_sol = pickle.load(file)
#
#     solve_dict = {}
#     for i in range(0, 100):
#         print(i)
#
#         (
#             model_spec_init_dict,
#             random_model_params_df,
#             exog_educ_shares,
#             exog_child_age_shares,
#             exog_partner_shares,
#             exog_exper_shares_pt,
#             exog_exper_shares_ft,
#             exog_child_info,
#             exog_partner_arrival_info,
#             exog_partner_separation_info,
#             expected_df_sim_func,
#         ) = tests_sim_func[i]
#
#         (
#             model_spec_init_dict,
#             random_model_params_df,
#             exog_educ_shares,
#             exog_child_age_shares,
#             exog_partner_shares,
#             exog_exper_shares_pt,
#             exog_exper_shares_ft,
#             exog_child_info,
#             exog_partner_arrival_info,
#             exog_partner_separation_info,
#             expected_df_sim_sol,
#         ) = tests_sim_sol[i]
#
#         #
#         #
#         #
#         # exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
#         # exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
#         # exog_child_info.to_pickle("test.soepy.child.pkl")
#         # exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
#         # exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
#         # exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
#         # exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
#         # exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")
#         #
#         # model_params_df, model_params = read_model_params_init(random_model_params_df)
#         # model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)
#         #
#         # prob_educ_level = gen_prob_educ_level_vector(model_spec)
#         # prob_child_age = gen_prob_child_init_age_vector(model_spec)
#         # prob_partner_present = gen_prob_partner_present_vector(model_spec)
#         # prob_exp_ft = gen_prob_init_exp_vector(
#         #     model_spec, model_spec.ft_exp_shares_file_name
#         # )
#         # prob_exp_pt = gen_prob_init_exp_vector(
#         #     model_spec, model_spec.pt_exp_shares_file_name
#         # )
#         # prob_child = gen_prob_child_vector(model_spec)
#         # prob_partner_arrival = gen_prob_partner_arrival(model_spec)
#         # prob_partner_separation = gen_prob_partner_separation(model_spec)
#         #
#         # solve_dict[i] = {}
#         # # Solve
#         # (
#         #     solve_dict[i]["states"],
#         #     solve_dict[i]["indexer"],
#         #     solve_dict[i]["covariates"],
#         #     solve_dict[i]["non_employment_consumption_resources"],
#         #     solve_dict[i]["emaxs"],
#         #     solve_dict[i]["child_age_update_rule"],
#         #     solve_dict[i]["deductions_spec"],
#         #     solve_dict[i]["income_tax_spec"],
#         # ) = pyth_solve(
#         #     model_params,
#         #     model_spec,
#         #     prob_child,
#         #     prob_partner_arrival,
#         #     prob_partner_separation,
#         #     is_expected=False,
#         # )
#         # # Simulate
#         # calculated_df = pyth_simulate(
#         #     model_params,
#         #     model_spec,
#         #     solve_dict[i]["states"],
#         #     solve_dict[i]["indexer"],
#         #     solve_dict[i]["emaxs"],
#         #     solve_dict[i]["covariates"],
#         #     solve_dict[i]["non_employment_consumption_resources"],
#         #     solve_dict[i]["deductions_spec"],
#         #     solve_dict[i]["income_tax_spec"],
#         #     solve_dict[i]["child_age_update_rule"],
#         #     prob_educ_level,
#         #     prob_child_age,
#         #     prob_partner_present,
#         #     prob_exp_ft,
#         #     prob_exp_pt,
#         #     prob_child,
#         #     prob_partner_arrival,
#         #     prob_partner_separation,
#         #     is_expected=False,
#         # )
#
#         unified_vault[i] = (
#             model_spec_init_dict,
#             random_model_params_df,
#             exog_educ_shares,
#             exog_child_age_shares,
#             exog_partner_shares,
#             exog_exper_shares_pt,
#             exog_exper_shares_ft,
#             exog_child_info,
#             exog_partner_arrival_info,
#             exog_partner_separation_info,
#             expected_df_sim_func,
#             expected_df_sim_sol
#         )
#
#     new_vault = TEST_RESOURCES_DIR / "regression_vault_new.soepy.pkl"
#
#     with open(new_vault, "wb") as file:
#         pickle.dump(unified_vault, file)
#
#     # new_vault = TEST_RESOURCES_DIR / "regression_sol_vault_new.soepy.pkl"
#     #
#     # with open(new_vault, "wb") as file:
#     #     pickle.dump(solve_dict, file)
#
#     cleanup()
#
#
# save_new_fault()
