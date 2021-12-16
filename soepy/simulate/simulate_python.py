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
from soepy.solve.solve_python import pyth_solve


def simulate(model_params_init_file_name, model_spec_init_file_name, is_expected=True):
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
    prob_partner_arrival = gen_prob_partner_arrival(model_spec)
    prob_partner_separation = gen_prob_partner_separation(model_spec)

    # Obtain model solution
    (
        states,
        indexer,
        covariates,
        non_employment_consumption_resources,
        emaxs,
        child_age_update_rule,
        deductions_spec,
    ) = pyth_solve(
        model_params,
        model_spec,
        prob_child,
        prob_partner_arrival,
        prob_partner_separation,
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
        deductions_spec,
        model_spec.tax_params,
        child_age_update_rule,
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

    return df


def prepare_simulation_solution(
    model_params_init_file_name, model_spec_init_file_name, is_expected=True
):

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
    prob_partner_arrival = gen_prob_partner_arrival(model_spec)
    prob_partner_separation = gen_prob_partner_separation(model_spec)

    # Obtain model solution
    (
        states,
        indexer,
        covariates,
        non_employment_consumption_resources,
        emaxs,
        child_age_update_rule,
        deductions_spec,
    ) = pyth_solve(
        model_params,
        model_spec,
        prob_child,
        prob_partner_arrival,
        prob_partner_separation,
        is_expected,
    )

    sol_dict = {
        "model_params": model_params,
        "model_spec": model_spec,
        "states": states,
        "indexer": indexer,
        "emaxs": emaxs,
        "covariates": covariates,
        "non_employment_consumption_resources": non_employment_consumption_resources,
        "deductions_spec": deductions_spec,
        "tax_params": model_spec.tax_params,
        "child_age_update_rule": child_age_update_rule,
        "prob_educ_level": prob_educ_level,
        "prob_child_age": prob_child_age,
        "prob_partner_present": prob_partner_present,
        "prob_exp_ft": prob_exp_ft,
        "prob_exp_pt": prob_exp_pt,
        "prob_child": prob_child,
        "prob_partner_arrival": prob_partner_arrival,
        "prob_partner_separation": prob_partner_separation,
        "is_expected": False,
    }

    return sol_dict
