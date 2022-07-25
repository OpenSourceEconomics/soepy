from functools import partial

from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.exogenous_processes.experience import gen_prob_init_exp_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.exogenous_processes.partner import gen_prob_partner_present_vector
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.simulate.simulate_auxiliary import pyth_simulate
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve


def simulate(
    model_params_init_file_name,
    model_spec_init_file_name,
    is_expected=True,
    data_sparse=False,
):
    """Create a data frame of individuals' simulated experiences."""

    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)

    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Get information concerning exogenous processes
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

    # Create state space
    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    # Obtain model solution
    non_employment_consumption_resources, emaxs = pyth_solve(
        states,
        covariates,
        child_state_indexes,
        model_params,
        model_spec,
        prob_child,
        prob_partner,
        is_expected,
    )

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
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
        data_sparse=data_sparse,
    )

    return df


def get_simulate_func(
    model_params_init_file_name,
    model_spec_init_file_name,
    is_expected=True,
    data_sparse=False,
):
    """Create the simulation function, such that the state space creation is already
    done ."""

    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)

    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Get information concerning exogenous processes
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

    # Create state space
    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec)

    partial_simulate = partial(
        partiable_simulate,
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
        prob_educ_level,
        prob_child_age,
        prob_partner_present,
        prob_exp_ft,
        prob_exp_pt,
        prob_child,
        prob_partner,
        is_expected,
        data_sparse,
    )
    return partial_simulate


def partiable_simulate(
    states,
    indexer,
    covariates,
    child_age_update_rule,
    child_state_indexes,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_ft,
    prob_exp_pt,
    prob_child,
    prob_partner,
    is_expected,
    data_sparse,
    model_params_init_file_name,
    model_spec_init_file_name,
):
    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)

    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Obtain model solution
    non_employment_consumption_resources, emaxs = pyth_solve(
        states,
        covariates,
        child_state_indexes,
        model_params,
        model_spec,
        prob_child,
        prob_partner,
        is_expected,
    )

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
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
        data_sparse=data_sparse,
    )

    return df
