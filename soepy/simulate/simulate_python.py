import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.initial_states import validate_initial_states
from soepy.simulate.simulate_auxiliary import JAX_SIM_OUTPUTS
from soepy.simulate.simulate_auxiliary import draw_simulation_randomness
from soepy.simulate.simulate_auxiliary import initial_states_to_jax_array
from soepy.simulate.simulate_auxiliary import jax_simulate_core
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import get_solve_function


def simulate(
    model_params_init_file_name,
    model_spec_init_file_name,
    *,
    initial_states,
    biased_exp=True,
    data_sparse=False,
):
    """Simulate a dataset given init specs."""

    simulate_func = get_simulate_func(
        model_params_init_file_name=model_params_init_file_name,
        model_spec_init_file_name=model_spec_init_file_name,
        initial_states=initial_states,
        biased_exp=biased_exp,
        data_sparse=data_sparse,
    )

    return simulate_func(
        model_params_init_file_name,
        model_spec_init_file_name,
    )


def get_simulate_func(
    model_params_init_file_name,
    model_spec_init_file_name,
    initial_states,
    biased_exp=True,
    data_sparse=False,
):
    """Create a simulation function with cached state space objects."""

    model_params_df, model_params = read_model_params_init(model_params_init_file_name)
    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    solve_func = get_solve_function(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=biased_exp,
    )

    initial_states_validated = validate_initial_states(initial_states, model_spec)

    def simulate_func(
        model_params_init_file_name_inner,
        model_spec_init_file_name_inner,
    ):
        model_params_df_inner, model_params_inner = read_model_params_init(
            model_params_init_file_name_inner
        )
        model_spec_inner = read_model_spec_init(
            model_spec_init_file_name_inner,
            model_params_df_inner,
        )

        non_consumption_utilities, emaxs = solve_func(model_params_inner)

        df = pyth_simulate(
            model_params=model_params_inner,
            model_spec=model_spec_inner,
            states=states,
            indexer=indexer,
            emaxs=emaxs,
            covariates=covariates,
            non_consumption_utilities=non_consumption_utilities,
            child_age_update_rule=child_age_update_rule,
            prob_child=prob_child,
            prob_partner=prob_partner,
            biased_exp=False,
            initial_states=initial_states_validated,
            data_sparse=data_sparse,
        ).set_index(["Identifier", "Period"])

        return df

    return simulate_func


def get_solve_and_simulate_func(
    model_params_init_file_name,
    model_spec_init_file_name,
    initial_states,
    biased_exp=True,
):
    """Create a JAX-backed simulation function with cached state space objects."""

    model_params_df, _ = read_model_params_init(model_params_init_file_name)
    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    solve_func = get_solve_function(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=biased_exp,
    )

    initial_states_validated = validate_initial_states(initial_states, model_spec)

    states = jnp.asarray(states)
    indexer = jnp.asarray(indexer)
    covariates = jnp.asarray(covariates)
    child_age_update_rule = jnp.asarray(child_age_update_rule)
    prob_child = jnp.asarray(prob_child)
    prob_partner = jnp.asarray(prob_partner)
    initial_states_array = initial_states_to_jax_array(initial_states_validated)

    def make_jitted_simulate(model_spec_inner):
        def simulate_core(
            model_params_inner,
            emaxs,
            non_consumption_utilities,
            type_uniforms,
            wage_draws,
            taste_shocks_standard,
            child_uniforms,
            partner_arrival_uniforms,
            partner_separation_uniforms,
        ):
            return jax_simulate_core(
                model_params=model_params_inner,
                model_spec=model_spec_inner,
                states=states,
                indexer=indexer,
                emaxs=emaxs,
                covariates=covariates,
                non_consumption_utilities=non_consumption_utilities,
                child_age_update_rule=child_age_update_rule,
                prob_child=prob_child,
                prob_partner=prob_partner,
                initial_states_array=initial_states_array,
                type_uniforms=type_uniforms,
                wage_draws=wage_draws,
                taste_shocks_standard=taste_shocks_standard,
                child_uniforms=child_uniforms,
                partner_arrival_uniforms=partner_arrival_uniforms,
                partner_separation_uniforms=partner_separation_uniforms,
                biased_exp=False,
            )

        return jax.jit(simulate_core)

    simulate_jit = make_jitted_simulate(model_spec)

    def uses_factory_spec(model_spec_init_file_name_inner):
        if model_spec_init_file_name_inner is model_spec_init_file_name:
            return True
        if isinstance(model_spec_init_file_name_inner, str) and isinstance(
            model_spec_init_file_name,
            str,
        ):
            return model_spec_init_file_name_inner == model_spec_init_file_name
        return False

    def simulate_func(
        model_params_init_file_name_inner,
        model_spec_init_file_name_inner,
    ):
        model_params_df_inner, model_params_inner = read_model_params_init(
            model_params_init_file_name_inner
        )
        model_spec_inner = read_model_spec_init(
            model_spec_init_file_name_inner,
            model_params_df_inner,
        )

        non_consumption_utilities, emaxs = solve_func(model_params_inner)
        random_draws = draw_simulation_randomness(
            model_spec=model_spec_inner,
            num_agents_sim=len(initial_states_validated),
        )

        simulate_jit_inner = (
            simulate_jit
            if uses_factory_spec(model_spec_init_file_name_inner)
            else make_jitted_simulate(model_spec_inner)
        )
        out = simulate_jit_inner(
            model_params_inner,
            emaxs,
            non_consumption_utilities,
            jnp.asarray(random_draws["type_uniforms"]),
            jnp.asarray(random_draws["wage_draws"]),
            jnp.asarray(random_draws["taste_shocks_standard"]),
            jnp.asarray(random_draws["child_uniforms"]),
            jnp.asarray(random_draws["partner_arrival_uniforms"]),
            jnp.asarray(random_draws["partner_separation_uniforms"]),
        )
        return jax_simulation_output_to_dataframe(out, JAX_SIM_OUTPUTS)

    return simulate_func


def jax_simulation_output_to_dataframe(out, columns):
    data = np.asarray(out["data"]).reshape(-1, len(columns))
    active = np.asarray(out["active"]).reshape(-1)
    df = pd.DataFrame(data[active], columns=columns)

    int_columns = [
        "Identifier",
        "Period",
        "Education_Level",
        "Lagged_Choice",
        "Experience_Part_Time",
        "Experience_Full_Time",
        "Type",
        "Age_Youngest_Child",
        "Partner_Indicator",
        "Choice",
    ]
    df[int_columns] = df[int_columns].astype(int)
    return df.set_index(["Identifier", "Period"])


def partiable_simulate(
    solve_func,
    states,
    indexer,
    covariates,
    child_age_update_rule,
    prob_child,
    prob_partner,
    data_sparse,
    model_params_init_file_name,
    model_spec_init_file_name,
    *,
    initial_states,
):
    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)

    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Obtain model solution
    non_consumption_utilities, emaxs = solve_func(model_params)

    # Simulate agents experiences according to parameters in the model specification
    initial_states_validated = validate_initial_states(initial_states, model_spec)
    df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
        non_consumption_utilities,
        child_age_update_rule,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=False,
        initial_states=initial_states_validated,
        data_sparse=data_sparse,
    ).set_index(["Identifier", "Period"])

    return df
