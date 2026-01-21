import pickle

import jax.numpy as jnp

from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.test.resources.aux_funcs import cleanup


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

        exog_educ_shares.to_pickle("test.soepy.educ.shares.pkl")
        exog_child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")
        exog_child_info.to_pickle("test.soepy.child.pkl")
        exog_partner_shares.to_pickle("test.soepy.partner.shares.pkl")
        exog_exper_shares_pt.to_pickle("test.soepy.pt.exp.shares.pkl")
        exog_exper_shares_ft.to_pickle("test.soepy.ft.exp.shares.pkl")
        exog_partner_arrival_info.to_pickle("test.soepy.partner.arrival.pkl")
        exog_partner_separation_info.to_pickle("test.soepy.partner.separation.pkl")

        # Add new inputs for continuous experience model
        exp_grid = jnp.linspace(0.0, 1.0, 10)
        model_spec_init_dict["exp_grid"] = exp_grid

        for old_category, new_category, old_param_name, new_param_name in [
            ("exp_returns_f", "exp_return", "gamma_f", "gamma_1"),
            ("exp_returns_p", "exp_increase_p", "gamma_p", "gamma_p"),
            (
                "exp_returns_p_bias",
                "exp_increase_p_bias",
                "gamma_p_bias",
                "gamma_p_bias",
            ),
        ]:
            for educ_ind, educ_type in enumerate(["low", "middle", "high"]):
                random_model_params_df.loc[
                    (new_category, f"{new_param_name}_{educ_type}"), "value"
                ] = random_model_params_df.loc[
                    (old_category, f"{old_param_name}_{educ_type}"), "value"
                ]
                # Delete old entry
                random_model_params_df = random_model_params_df.drop(
                    index=(old_category, f"{old_param_name}_{educ_type}")
                )

        # Sort index after modifications
        random_model_params_df = random_model_params_df.sort_index()

        calculated_df_sim = simulate(random_model_params_df, model_spec_init_dict)
        unbiased_calc_df = simulate(
            random_model_params_df, model_spec_init_dict, is_expected=False
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
            calculated_df_sim.reset_index().sum(axis=0),
            unbiased_calc_df.reset_index().sum(axis=0),
        )

    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    cleanup(options="regression")


# update_sim_objectes()
