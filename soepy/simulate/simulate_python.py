from functools import partial

from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import get_solve_function
from soepy.simulate.initial_states import validate_initial_states


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
        biased_exp=biased_exp,
        data_sparse=data_sparse,
    )

    return simulate_func(
        model_params_init_file_name,
        model_spec_init_file_name,
        initial_states=initial_states,
    )


def get_simulate_func(
    model_params_init_file_name,
    model_spec_init_file_name,
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


    def simulate_func(
        model_params_init_file_name_inner,
        model_spec_init_file_name_inner,
        *,
        initial_states,
    ):
        model_params_df_inner, model_params_inner = read_model_params_init(
            model_params_init_file_name_inner
        )
        model_spec_inner = read_model_spec_init(
            model_spec_init_file_name_inner,
            model_params_df_inner,
        )

        non_consumption_utilities, emaxs = solve_func(model_params_inner)

        initial_states_validated = validate_initial_states(
            initial_states, model_spec_inner
        )
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
