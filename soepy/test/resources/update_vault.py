import pickle
import random

import pandas as pd

from development.tests.auxiliary.auxiliary import cleanup
from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_vector
from soepy.exogenous_processes.partner import gen_prob_partner_arrival
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.exogenous_processes.partner import gen_prob_partner_separation
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.solve_python import pyth_solve


def update_solve_objectes():
    vault_file = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    vault = {}
    with open(vault_file, "rb") as file:
        tests_sim_func = pickle.load(file)

    solve_dict = {}
    for i in range(0, 100):
        print(i)

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
        ) = tests_sim_func[i]

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"] = {}
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["under_3"] = [
            219,
            381,
        ]
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["3_to_6"] = [
            122,
            128,
        ]

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
        ) = pyth_solve(
            model_params,
            model_spec,
            prob_child,
            prob_partner_arrival,
            prob_partner_separation,
            is_expected=False,
        )
        # Simulate
        calculated_df_sim_sol = pyth_simulate(
            model_params,
            model_spec,
            solve_dict[i]["states"],
            solve_dict[i]["indexer"],
            solve_dict[i]["emaxs"],
            solve_dict[i]["covariates"],
            solve_dict[i]["non_employment_consumption_resources"],
            solve_dict[i]["deductions_spec"],
            model_spec.tax_params,
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

        vault[i] = (
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
            calculated_df_sim_sol,
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    cleanup()


def update_sim_objectes():
    vault_file = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    vault = {}
    with open(vault_file, "rb") as file:
        tests_sim_func = pickle.load(file)

    for i in range(0, 100):
        print(i)

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
        ) = tests_sim_func[i]

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"] = {}
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["under_3"] = [
            219,
            381,
        ]
        model_spec_init_dict["TAXES_TRANSFERS"]["child_care_costs"]["3_to_6"] = [
            122,
            128,
        ]

        calculated_df_sim = simulate(random_model_params_df, model_spec_init_dict)

        vault[i] = (
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
            calculated_df_sim,
            expected_df_sim_sol,
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)


update_sim_objectes()
update_solve_objectes()
