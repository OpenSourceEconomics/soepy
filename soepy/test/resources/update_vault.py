import pickle

from development.tests.auxiliary.auxiliary import cleanup
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR


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
            expected_df,
            expected_df_unbiased,
        ) = tests_sim_func[i]

        random_model_params_df.loc[
            ("disutil_work", "child_0_2_f_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_0_2_p_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_f_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_p_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_f_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_p_educ_high"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_0_2_f_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_0_2_p_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_f_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_p_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_f_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_p_educ_middle"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_0_2_f_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_0_2_p_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_0_2_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_f_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_3_5_p_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_3_5_p"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_f_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_f"), :]
        random_model_params_df.loc[
            ("disutil_work", "child_6_10_p_educ_low"), :
        ] = random_model_params_df.loc[("disutil_work", "child_6_10_p"), :]

        random_model_params_df.drop(index=("disutil_work", "child_0_2_f"), inplace=True)
        random_model_params_df.drop(index=("disutil_work", "child_0_2_p"), inplace=True)
        random_model_params_df.drop(index=("disutil_work", "child_3_5_f"), inplace=True)
        random_model_params_df.drop(index=("disutil_work", "child_3_5_p"), inplace=True)
        random_model_params_df.drop(
            index=("disutil_work", "child_6_10_f"), inplace=True
        )
        random_model_params_df.drop(
            index=("disutil_work", "child_6_10_p"), inplace=True
        )

        random_model_params_df.sort_index(inplace=True)

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

        # calculated_df_sim = simulate(random_model_params_df, model_spec_init_dict)
        # unbiased_calc_df = simulate(
        #     random_model_params_df, model_spec_init_dict, is_expected=False
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
            expected_df,
            expected_df_unbiased,
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    cleanup(options="regression")


# update_sim_objectes()
