import pickle

import numpy as np

from development.tests.auxiliary.auxiliary import cleanup
from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve


def update_solve_objectes():
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

        random_model_params_df.loc[
            ("disutil_work", f"child_0_2_p"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_02_p"), "value"]
        random_model_params_df.loc[
            ("disutil_work", f"child_0_2_f"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_02_f"), "value"]

        random_model_params_df.loc[
            ("disutil_work", f"child_3_5_p"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_35_p"), "value"]
        random_model_params_df.loc[
            ("disutil_work", f"child_3_5_f"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_35_f"), "value"]

        random_model_params_df.loc[
            ("disutil_work", f"child_6_10_p"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_6orolder_p"), "value"]
        random_model_params_df.loc[
            ("disutil_work", f"child_6_10_f"), "value"
        ] = random_model_params_df.loc[("disutil_work", f"child_6orolder_f"), "value"]

        random_model_params_df.loc[("disutil_work", f"child_11_older_p"), "value"] = 0
        random_model_params_df.loc[("disutil_work", f"child_11_older_f"), "value"] = 0

        for param in [
            "child_02_p",
            "child_02_f",
            "child_35_p",
            "child_35_f",
            "child_6orolder_p",
            "child_6orolder_f",
        ]:
            random_model_params_df.drop(("disutil_work", param), inplace=True)

        random_model_params_df.sort_index(inplace=True)

        # exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        # exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        # exog_child_info.to_pickle("test.soepy.child.pkl")
        # exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        # exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        # exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        # exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        # exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")
        #
        # model_params_df, model_params = read_model_params_init(random_model_params_df)
        # model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)
        #
        # prob_educ_level = gen_prob_educ_level_vector(model_spec)
        # prob_child_age = gen_prob_child_init_age_vector(model_spec)
        # prob_partner_present = gen_prob_partner_present_vector(model_spec)
        # prob_exp_ft = gen_prob_init_exp_vector(
        #     model_spec, model_spec.ft_exp_shares_file_name
        # )
        # prob_exp_pt = gen_prob_init_exp_vector(
        #     model_spec, model_spec.pt_exp_shares_file_name
        # )
        # prob_child = gen_prob_child_vector(model_spec)
        # prob_partner = gen_prob_partner(model_spec)
        #
        # # Create state space
        # (
        #     states,
        #     indexer,
        #     covariates,
        #     child_age_update_rule,
        #     child_state_indexes,
        # ) = create_state_space_objects(model_spec)
        #
        # # Obtain model solution
        # non_employment_consumption_resources, emaxs = pyth_solve(
        #     states,
        #     covariates,
        #     child_state_indexes,
        #     model_params,
        #     model_spec,
        #     prob_child,
        #     prob_partner,
        #     False,
        # )
        # # Simulate
        # calculated_df_sim_sol = pyth_simulate(
        #     model_params,
        #     model_spec,
        #     states,
        #     indexer,
        #     emaxs,
        #     covariates,
        #     non_employment_consumption_resources,
        #     child_age_update_rule,
        #     prob_educ_level,
        #     prob_child_age,
        #     prob_partner_present,
        #     prob_exp_ft,
        #     prob_exp_pt,
        #     prob_child,
        #     prob_partner,
        #     is_expected=False,
        # )

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
            expected_df_sim_sol,
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    cleanup(options="regression")


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
            calculated_df_sim.sum(axis=0),
            expected_df_sim_sol,
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    cleanup(options="regression")


# update_sim_objectes()
update_solve_objectes()
