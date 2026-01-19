import numpy as np
import pandas as pd

from soepy.exogenous_processes.determine_lagged_choice import lagged_choice_initial
from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import draw_zero_one_distributed_shocks
from soepy.shared.shared_constants import HOURS
from soepy.shared.shared_constants import NUM_CHOICES
from soepy.shared.wages import calculate_log_wage
from soepy.simulate.constants_sim import DATA_FORMATS_SIM
from soepy.simulate.constants_sim import DATA_FORMATS_SPARSE
from soepy.simulate.constants_sim import DATA_LABLES_SIM
from soepy.simulate.constants_sim import IDX_STATES_DATA_SPARSE
from soepy.simulate.constants_sim import LABELS_DATA_SPARSE
from soepy.simulate.income_sim import calculate_employment_consumption_resources


IDX_ID = 0
IDX_PERIOD = 1
IDX_EDUC = 2
IDX_LAGGED = 3
IDX_STOCK = 4
IDX_TYPE = 5
IDX_CHILD_AGE = 6
IDX_PARTNER = 7


def pyth_simulate(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_consumption_utilities,
    child_age_update_rule,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_years,
    prob_child,
    prob_partner,
    is_expected,
    data_sparse=False,
):
    """Simulate agent histories under the continuous-experience model."""

    np.random.seed(model_spec.seed_sim)

    exp_grid_size = int(getattr(model_spec, "experience_grid_points", 10))
    exp_grid = np.linspace(0.0, 1.0, exp_grid_size)

    emaxs = np.asarray(emaxs)
    non_consumption_utilities = np.asarray(non_consumption_utilities)

    (
        initial_states,
        draws_sim,
        log_wage_grid,
        non_emp_resources_grid,
    ) = prepare_simulation_data(
        model_params=model_params,
        model_spec=model_spec,
        prob_educ_level=prob_educ_level,
        prob_child_age=prob_child_age,
        prob_partner_present=prob_partner_present,
        prob_exp_years=prob_exp_years,
        states=states,
        covariates=covariates,
        tax_splitting=model_spec.tax_splitting,
        is_expected=is_expected,
        exp_grid=exp_grid,
    )

    data = simulate_agents_over_periods(
        model_spec=model_spec,
        state_space=states,
        indexer=indexer,
        covariates=covariates,
        emaxs=emaxs,
        non_consumption_utilities=non_consumption_utilities,
        child_age_update_rule=child_age_update_rule,
        prob_child=prob_child,
        prob_partner=prob_partner,
        draws_sim=draws_sim,
        initial_states=initial_states,
        log_wage_grid=log_wage_grid,
        non_emp_resources_grid=non_emp_resources_grid,
        exp_grid=exp_grid,
        model_params=model_params,
        is_expected=is_expected,
        data_sparse=data_sparse,
    )

    if data_sparse:
        dataset = pd.DataFrame(np.vstack(data), columns=LABELS_DATA_SPARSE).astype(
            DATA_FORMATS_SPARSE
        )
    else:
        dataset = pd.DataFrame(np.vstack(data), columns=DATA_LABLES_SIM).astype(
            DATA_FORMATS_SIM
        )

    dataset.loc[dataset["Choice"] == 0, "Wage_Observed"] = np.nan

    return dataset


def _get_pt_increment(model_params, model_spec, educ_level, is_expected):
    if is_expected:
        inc = model_params.gamma_p_bias
    else:
        # Default to 1.0 if not configured in legacy specs.
        inc = getattr(model_spec, "pt_exp_ratio", 1.0)

    if hasattr(inc, "__len__"):
        return np.asarray(inc)[educ_level]

    return np.ones_like(educ_level, dtype=float) * float(inc)


def _max_exp_years(period, init_exp_max, pt_increment):
    return init_exp_max + np.maximum(period, period * pt_increment)


def _next_stock(stock, period, init_exp_max, pt_increment, choice):
    max_years_t = _max_exp_years(period, init_exp_max, pt_increment)
    exp_years = stock * max_years_t

    exp_years_next = exp_years + (choice == 2) * 1.0 + (choice == 1) * pt_increment

    max_years_tp1 = _max_exp_years(period + 1, init_exp_max, pt_increment)
    denom = np.where(max_years_tp1 > 0, max_years_tp1, 1.0)

    stock_next = exp_years_next / denom
    return np.clip(stock_next, 0.0, 1.0)


def _interp_uniform_grid(values, stock):
    """Interpolate along the last axis on a uniform [0,1] grid.

    Parameters
    ----------
    values : np.ndarray
        Shape (..., n_grid)
    stock : np.ndarray
        Shape (...) matching the leading dimensions of values.
    """

    n_grid = values.shape[-1]
    u = np.clip(stock, 0.0, 1.0) * (n_grid - 1)

    idx_lo = np.floor(u).astype(int)
    idx_hi = np.minimum(idx_lo + 1, n_grid - 1)
    w = u - idx_lo

    take_lo = np.take_along_axis(values, idx_lo[..., None], axis=-1)[..., 0]
    take_hi = np.take_along_axis(values, idx_hi[..., None], axis=-1)[..., 0]

    return take_lo + w * (take_hi - take_lo)


def prepare_simulation_data(
    model_params,
    model_spec,
    prob_educ_level,
    prob_child_age,
    prob_partner_present,
    prob_exp_years,
    states,
    covariates,
    tax_splitting,
    is_expected,
    exp_grid,
):
    """Draw initial conditions and precompute grid objects for simulation."""

    initial_educ_level = np.random.choice(
        model_spec.num_educ_levels, model_spec.num_agents_sim, p=prob_educ_level
    )

    initial_period = np.asarray(model_spec.educ_years)[initial_educ_level]

    initial_child_age = np.full(model_spec.num_agents_sim, np.nan)
    initial_partner = np.full(model_spec.num_agents_sim, np.nan)
    initial_exp_years = np.full(model_spec.num_agents_sim, np.nan)

    for educ_level in range(model_spec.num_educ_levels):
        mask = initial_educ_level == educ_level

        initial_child_age[mask] = np.random.choice(
            list(range(-1, model_spec.child_age_init_max + 1)),
            mask.sum(),
            p=prob_child_age[educ_level],
        )

        initial_partner[mask] = np.random.binomial(
            size=mask.sum(),
            n=1,
            p=prob_partner_present[educ_level],
        )

        initial_exp_years[mask] = np.random.choice(
            list(range(0, 2 * model_spec.init_exp_max + 1)),
            mask.sum(),
            p=prob_exp_years[educ_level],
        )

    lagged_choice = lagged_choice_initial(initial_exp_years)

    type_ = np.random.choice(
        np.arange(model_spec.num_types),
        model_spec.num_agents_sim,
        p=model_params.type_shares,
    )

    draws_sim = draw_zero_one_distributed_shocks(
        model_spec.seed_sim, model_spec.num_periods, model_spec.num_agents_sim
    )
    draws_sim = draws_sim * float(model_params.shock_sd)

    pt_increment_init = _get_pt_increment(
        model_params=model_params,
        model_spec=model_spec,
        educ_level=initial_educ_level,
        is_expected=is_expected,
    )

    max_years_init = _max_exp_years(
        period=initial_period,
        init_exp_max=model_spec.init_exp_max,
        pt_increment=pt_increment_init,
    )
    denom = np.where(max_years_init > 0, max_years_init, 1.0)
    initial_stock = np.clip(initial_exp_years / denom, 0.0, 1.0)

    initial_states = pd.DataFrame(
        {
            "Identifier": np.arange(model_spec.num_agents_sim, dtype=int),
            "Period": initial_period.astype(int),
            "Education_Level": initial_educ_level.astype(int),
            "Lagged_Choice": lagged_choice.astype(int),
            "Experience_Stock": initial_stock.astype(float),
            "Type": type_.astype(int),
            "Age_Youngest_Child": initial_child_age.astype(int),
            "Partner_Indicator": initial_partner.astype(int),
        }
    )

    pt_increment_grid = _get_pt_increment(
        model_params=model_params,
        model_spec=model_spec,
        educ_level=states[:, 1],
        is_expected=is_expected,
    )

    max_years = _max_exp_years(
        period=states[:, 0],
        init_exp_max=model_spec.init_exp_max,
        pt_increment=pt_increment_grid,
    )

    exp_years = exp_grid[None, :] * max_years[:, None]

    log_wage_grid = np.asarray(
        calculate_log_wage(
            model_params=model_params, states=states, exp_years=exp_years
        )
    )

    non_emp_resources_grid = np.asarray(
        calculate_non_employment_consumption_resources(
            deductions_spec=model_spec.ssc_deductions,
            income_tax_spec=model_spec.tax_params,
            model_spec=model_spec,
            states=states,
            log_wage_systematic=log_wage_grid,
            male_wage=covariates[:, 1],
            child_benefits=covariates[:, 3],
            tax_splitting=tax_splitting,
            hours=HOURS,
        )
    )

    return initial_states, draws_sim, log_wage_grid, non_emp_resources_grid


def simulate_agents_over_periods(
    model_spec,
    state_space,
    indexer,
    covariates,
    emaxs,
    non_consumption_utilities,
    child_age_update_rule,
    prob_child,
    prob_partner,
    draws_sim,
    initial_states,
    log_wage_grid,
    non_emp_resources_grid,
    exp_grid,
    model_params,
    is_expected,
    data_sparse,
):
    data = []

    current_states = np.empty((0, 8), dtype=float)

    for period in range(model_spec.num_periods):
        entrants = initial_states.loc[initial_states.Period.eq(period)].to_numpy()
        if entrants.size:
            current_states = np.vstack((current_states, entrants))

        if current_states.size == 0:
            continue

        age_idx = np.where(
            current_states[:, IDX_CHILD_AGE] == -1,
            indexer.shape[4] - 1,
            current_states[:, IDX_CHILD_AGE],
        )

        idx = indexer[
            current_states[:, IDX_PERIOD].astype(int),
            current_states[:, IDX_EDUC].astype(int),
            current_states[:, IDX_LAGGED].astype(int),
            current_states[:, IDX_TYPE].astype(int),
            age_idx.astype(int),
            current_states[:, IDX_PARTNER].astype(int),
        ]

        stock = current_states[:, IDX_STOCK].astype(float)

        log_wage_agents = _interp_uniform_grid(log_wage_grid[idx], stock)
        non_emp_resources_agents = _interp_uniform_grid(
            non_emp_resources_grid[idx], stock
        )

        cont_grid = emaxs[idx, :, :NUM_CHOICES]
        cont_grid = np.transpose(cont_grid, (0, 2, 1))
        continuation_values = _interp_uniform_grid(cont_grid, stock[:, None])

        non_cons_util_agents = non_consumption_utilities[idx]

        wages = np.exp(
            log_wage_agents + draws_sim[period, current_states[:, IDX_ID].astype(int)]
        )
        wages = wages * float(model_spec.elasticity_scale)

        female_income = wages[:, None] * HOURS[None, 1:]

        male_wage = covariates[idx][:, 1]
        child_benefits = covariates[idx][:, 3]
        equiv_scale = covariates[idx][:, 2]

        employment_resources = calculate_employment_consumption_resources(
            model_spec,
            female_income,
            male_wage,
            model_spec.tax_splitting,
        )

        if model_spec.parental_leave_regime == "erziehungsgeld":
            married = current_states[:, IDX_PARTNER] == 1
            baby_child = (current_states[:, IDX_CHILD_AGE] == 0) | (
                current_states[:, IDX_CHILD_AGE] == 1
            )

            erz = calc_erziehungsgeld(
                male_wage=male_wage,
                female_income=female_income[:, 0],
                married=married,
                baby_child=baby_child,
                erziehungsgeld_inc_single=model_spec.erziehungsgeld_income_threshold_single,
                erziehungsgeld_inc_married=model_spec.erziehungsgeld_income_threshold_married,
                erziehungsgeld=model_spec.erziehungsgeld,
            )
            employment_resources[:, 0] = employment_resources[:, 0] + erz

        employment_resources = employment_resources + child_benefits[:, None]

        child_care_costs = get_child_care_cost_for_choice(
            covariates[idx][:, 0].astype(float), model_spec.child_care_costs
        )
        employment_resources = employment_resources - child_care_costs

        consumption_resources = np.hstack(
            (non_emp_resources_agents[:, None], employment_resources)
        )
        consumption_resources = consumption_resources.clip(min=np.finfo(float).eps)

        flow_utilities = (
            (consumption_resources / equiv_scale[:, None]) ** float(model_params.mu)
            / float(model_params.mu)
            * non_cons_util_agents
        )

        value_functions = (
            flow_utilities + float(model_params.delta) * continuation_values
        )
        choice = np.argmax(value_functions, axis=1)

        if data_sparse:
            rows = np.column_stack(
                (
                    current_states[:, IDX_STATES_DATA_SPARSE].copy(),
                    choice,
                    wages,
                )
            )
        else:
            rows = np.column_stack(
                (
                    current_states.copy(),
                    choice,
                    log_wage_agents,
                    wages,
                    non_cons_util_agents,
                    flow_utilities,
                    continuation_values,
                    value_functions,
                    male_wage,
                )
            )

        data.append(rows)

        # --- exogenous updates
        child_current_age = current_states[:, IDX_CHILD_AGE]

        if period == model_spec.num_periods - 1:
            child_new_age = child_current_age
        elif period <= model_spec.last_child_bearing_period:
            kids_draw = np.random.binomial(
                size=current_states.shape[0],
                n=1,
                p=prob_child[period + 1, current_states[:, IDX_EDUC].astype(int)],
            )
            child_new_age = np.where(kids_draw == 0, child_age_update_rule[idx], 0)
        else:
            child_new_age = child_age_update_rule[idx]

        current_partner = current_states[:, IDX_PARTNER]
        new_partner = current_partner.copy()

        no_partner = current_partner == 0
        if no_partner.any():
            arr = np.random.binomial(
                size=no_partner.sum(),
                n=1,
                p=prob_partner[
                    period, current_states[no_partner, IDX_EDUC].astype(int), 0, 1
                ],
            )
            new_partner[no_partner] = arr

        has_partner = current_partner == 1
        if has_partner.any():
            sep = np.random.binomial(
                size=has_partner.sum(),
                n=1,
                p=prob_partner[
                    period, current_states[has_partner, IDX_EDUC].astype(int), 1, 0
                ],
            )
            new_partner[has_partner] = current_partner[has_partner] - sep

        # --- endogenous updates
        pt_increment = _get_pt_increment(
            model_params=model_params,
            model_spec=model_spec,
            educ_level=current_states[:, IDX_EDUC].astype(int),
            is_expected=is_expected,
        )
        current_states[:, IDX_STOCK] = _next_stock(
            stock=current_states[:, IDX_STOCK].astype(float),
            period=current_states[:, IDX_PERIOD].astype(int),
            init_exp_max=model_spec.init_exp_max,
            pt_increment=pt_increment,
            choice=choice,
        )

        current_states[:, IDX_PERIOD] = current_states[:, IDX_PERIOD] + 1
        current_states[:, IDX_LAGGED] = choice
        current_states[:, IDX_CHILD_AGE] = child_new_age
        current_states[:, IDX_PARTNER] = new_partner

    return data


def get_child_care_cost_for_choice(child_bins, child_care_costs):
    child_bins = child_bins.copy()
    child_bins[child_bins > 2] = 0

    child_costs = np.zeros((child_bins.shape[0], 2))
    for choice in range(2):
        for age_bin in range(1, 3):
            child_costs[child_bins == age_bin, choice] = child_care_costs[
                age_bin, choice
            ]
    return child_costs
