from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.children import gen_prob_child_init_age_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.exogenous_processes.education import gen_prob_educ_level_vector
from soepy.solve.solve_python import pyth_solve
from soepy.simulate.simulate_auxiliary import pyth_simulate


def simulate(model_params_init_file_name, model_spec_init_file_name, is_expected=True):
    """Create a data frame of individuals' simulated experiences."""

    # Read in model specification from yaml file
    model_params_df, model_params = read_model_params_init(model_params_init_file_name)
    model_spec = read_model_spec_init(model_spec_init_file_name, model_params_df)

    # Get information concerning exogenous processes
    prob_child = gen_prob_child_vector(model_spec)
    prob_partner = gen_prob_partner(model_spec)
    prob_educ_level = gen_prob_educ_level_vector(model_spec)
    prob_child_age = gen_prob_child_init_age_vector(model_spec)

    # Obtain model solution
    (
        states,
        indexer,
        covariates,
        budget_constraint_components,
        non_employment_benefits,
        emaxs,
        child_age_update_rule,
    ) = pyth_solve(model_params, model_spec, prob_child, prob_partner, is_expected)

    # Simulate agents experiences according to parameters in the model specification
    df = pyth_simulate(
        model_params,
        model_spec,
        states,
        indexer,
        emaxs,
        covariates,
        budget_constraint_components,
        non_employment_benefits,
        child_age_update_rule,
        prob_child,
        prob_partner,
        prob_educ_level,
        prob_child_age,
        is_expected=False,
    )

    return df
