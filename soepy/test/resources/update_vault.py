import pickle

import jax.numpy as jnp

from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_python import simulate
from soepy.soepy_config import TEST_RESOURCES_DIR
from soepy.test.resources.aux_funcs import cleanup
from soepy.test.resources.aux_funcs import move_initial_conditions
from soepy.test.resources.exogenous_processes import gen_prob_child_init_age_vector
from soepy.test.resources.exogenous_processes import gen_prob_educ_level_vector
from soepy.test.resources.exogenous_processes import gen_prob_init_exp_component_vector
from soepy.test.resources.exogenous_processes import gen_prob_partner_present_vector
from soepy.test.resources.initial_states import create_initial_states_from_probs


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

        # Sort index after modifications
        random_model_params_df = random_model_params_df.sort_index()

        model_params_df, model_params = read_model_params_init(random_model_params_df)
        model_spec_init_dict = move_initial_conditions(model_spec_init_dict)
        model_spec = read_model_spec_init(model_spec_init_dict, model_params_df)

        prob_educ_level = gen_prob_educ_level_vector(model_spec)
        prob_child_age = gen_prob_child_init_age_vector(model_spec)
        prob_partner_present = gen_prob_partner_present_vector(model_spec)
        prob_exp_pt = gen_prob_init_exp_component_vector(
            model_spec, model_spec.pt_exp_shares_file_name
        )
        prob_exp_ft = gen_prob_init_exp_component_vector(
            model_spec, model_spec.ft_exp_shares_file_name
        )

        random_model_params_df.loc[("exp_increase_p_mom", "gamma_p_mom"), "value"] = (
            -random_model_params_df.loc[("exp_increase_p", slice(None)), "value"].min()
            / 2
        )

        initial_states = create_initial_states_from_probs(
            model_spec=model_spec,
            prob_educ_level=prob_educ_level,
            prob_child_age=prob_child_age,
            prob_partner_present=prob_partner_present,
            prob_exp_pt=prob_exp_pt,
            prob_exp_ft=prob_exp_ft,
        )

        calculated_df_sim = simulate(
            random_model_params_df,
            model_spec_init_dict,
            initial_states=initial_states,
        )
        unbiased_calc_df = simulate(
            random_model_params_df,
            model_spec_init_dict,
            initial_states=initial_states,
            biased_exp=False,
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
