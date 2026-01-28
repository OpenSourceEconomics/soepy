import numpy as np

from soepy.exogenous_processes.children import gen_prob_child_vector
from soepy.exogenous_processes.partner import gen_prob_partner
from soepy.pre_processing.model_processing import read_model_params_init
from soepy.pre_processing.model_processing import read_model_spec_init
from soepy.shared.constants_and_indices import HOURS
from soepy.shared.experience_stock import get_pt_increment
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.tax_and_transfers_jax import calculate_net_income
from soepy.shared.wages import calculate_log_wage
from soepy.solve.create_state_space import create_state_space_objects
from soepy.solve.solve_python import pyth_solve
from soepy.test.random_init import random_init


def test_construct_emax_nonemployment_branch_matches_value_function():
    """Check a Bellman identity in the terminal period.

    In the last period, continuation values are zero and the expected max value equals
    the maximum over the three instantaneous utilities (N/PT/FT).

    This is a modernized replacement for the legacy discrete-experience
    `test_construction_emax.py`.
    """

    constr = {
        "AGENTS": 200,
        "PERIODS": 4,
        "EDUC_YEARS": [0, 0, 0],
        "CHILD_AGE_INIT_MAX": 1,
        "INIT_EXP_MAX": 1,
        "SEED_SIM": 123,
        "SEED_EMAX": 321,
        "NUM_DRAWS_EMAX": 10,
    }
    random_init(constr)

    model_params_df, model_params = read_model_params_init("test.soepy.pkl")
    model_spec = read_model_spec_init("test.soepy.yml", model_params_df)

    prob_child = gen_prob_child_vector(model_spec=model_spec)
    prob_partner = gen_prob_partner(model_spec=model_spec)

    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec=model_spec)

    non_consumption_utilities, emaxs = pyth_solve(
        states=states,
        covariates=covariates,
        child_state_indexes=child_state_indexes,
        model_params=model_params,
        model_spec=model_spec,
        prob_child=prob_child,
        prob_partner=prob_partner,
        biased_exp=False,
    )

    # Last period: continuation is zero, so max value is max over instant utilities.
    mask_last = states[:, 0] == (model_spec.num_periods - 1)
    states_last = states[mask_last]
    cov_last = covariates[mask_last]
    non_cons_last = np.asarray(non_consumption_utilities)[mask_last]
    emax_last = np.asarray(emaxs)[mask_last]

    # Non-employment resources at the experience grid points.
    pt_inc = get_pt_increment(
        model_params=model_params,
        educ_level=states_last[:, 1],
        child_age=states_last[:, 4],
        biased_exp=False,
    )
    exp_grid = np.asarray(model_spec.exp_grid)

    def log_wage_one_gridpoint(exp_stock):
        return np.asarray(
            calculate_log_wage(
                model_params=model_params,
                educ=states_last[:, 1],
                period=states_last[:, 0],
                init_exp_max=model_spec.init_exp_max,
                pt_increment=pt_inc,
                exp_stock=exp_stock,
            )
        )

    log_wage_grid = np.stack([log_wage_one_gridpoint(x) for x in exp_grid], axis=1)

    non_emp_resources_grid = calculate_non_employment_consumption_resources(
        deductions_spec=np.array(model_spec.ssc_deductions),
        income_tax_spec=model_spec.tax_params,
        model_spec=model_spec,
        states=states_last,
        log_wage_systematic=log_wage_grid,
        male_wage=cov_last[:, 1],
        child_benefits=cov_last[:, 3],
        tax_splitting=model_spec.tax_splitting,
        hours=HOURS,
    )

    # Instantaneous utility for N.
    mu = float(model_params.mu)
    equiv = cov_last[:, 2][:, None]
    cons_n = np.maximum(non_emp_resources_grid / equiv, 1e-14)
    util_n = (cons_n**mu / mu) * non_cons_last[:, 0][:, None]

    # Instantaneous utilities for work choices at each experience grid point.
    draw = 0.0
    male_wage = cov_last[:, 1][:, None]
    child_benefits = cov_last[:, 3][:, None]

    female_wage_pt = HOURS[1] * np.exp(log_wage_grid + draw)
    female_wage_ft = HOURS[2] * np.exp(log_wage_grid + draw)

    def util_work(female_wage, non_cons_col, child_care_choice_col):
        net = np.asarray(
            calculate_net_income(
                model_spec.tax_params,
                np.asarray(model_spec.ssc_deductions),
                female_wage,
                male_wage,
                model_spec.tax_splitting,
            )
        )
        # Child care costs depend on a binned child age (in covariates[:,0]).
        child_bin = cov_last[:, 0].astype(int)
        child_bin = np.where(child_bin > 2, 0, child_bin)
        child_costs = np.asarray(model_spec.child_care_costs)[
            child_bin, child_care_choice_col
        ]
        child_costs = child_costs[:, None]

        cons = np.maximum((net + child_benefits - child_costs) / equiv, 1e-14)
        return (cons**mu / mu) * non_cons_col[:, None]

    util_pt = util_work(female_wage_pt, non_cons_last[:, 1], 0)
    util_ft = util_work(female_wage_ft, non_cons_last[:, 2], 1)

    vmax = np.maximum(util_n, np.maximum(util_pt, util_ft))

    # Small numerical differences can occur due to mixing NumPy/JAX code paths.
    np.testing.assert_allclose(vmax, emax_last[:, :, 3], rtol=1e-6, atol=2e-2)
