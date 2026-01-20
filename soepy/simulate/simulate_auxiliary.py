import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.wages import calculate_log_wage
from soepy.simulate.constants_sim import DATA_FORMATS_SIM
from soepy.simulate.constants_sim import DATA_FORMATS_SPARSE
from soepy.simulate.constants_sim import DATA_LABLES_SIM
from soepy.simulate.constants_sim import IDX_STATES_DATA_SPARSE
from soepy.simulate.constants_sim import LABELS_DATA_SPARSE
from soepy.simulate.income_sim import calculate_employment_consumption_resources
from soepy.simulate.initial_states import prepare_simulation_data

IDX_PERIOD = 1
IDX_EDUC = 2
IDX_LAGGED = 3
IDX_EXP_PT = 4
IDX_EXP_FT = 5
IDX_EXP_STOCK = 6
IDX_TYPE = 7
IDX_CHILD_AGE = 8
IDX_PARTNER = 9


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
    prob_exp_pt,
    prob_exp_ft,
    prob_child,
    prob_partner,
    is_expected,
    data_sparse=False,
):
    """Simulate agent histories under the continuous-experience model."""

    np.random.seed(model_spec.seed_sim)

    exp_grid = np.asarray(model_spec.exp_grid)

    emaxs = np.asarray(emaxs)
    non_consumption_utilities = np.asarray(non_consumption_utilities)

    (initial_states, draws_sim,) = prepare_simulation_data(
        model_params=model_params,
        model_spec=model_spec,
        prob_educ_level=prob_educ_level,
        prob_child_age=prob_child_age,
        prob_partner_present=prob_partner_present,
        prob_exp_pt=prob_exp_pt,
        prob_exp_ft=prob_exp_ft,
        is_expected=is_expected,
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
    model_params,
    is_expected,
    data_sparse,
):
    data = []
    current_states = np.empty((0, 10), dtype=float)

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

        stock = current_states[:, IDX_EXP_STOCK].astype(float)
        breakpoint()

        # Interpolate continuation values on the experience grid.
        continuation_grid = emaxs[idx, :, :NUM_CHOICES]
        continuation_grid = np.transpose(continuation_grid, (0, 2, 1))
        continuation_values = _interp_uniform_grid(continuation_grid, stock[:, None])

        non_cons_util_agents = non_consumption_utilities[idx]

        log_wage_agents = calculate_log_wage(
            model_params=model_params,
            educ=state_space[idx],
            exp_stock=current_states[:, IDX_EXP_STOCK],
        )[:, 0]

        wages = np.exp(log_wage_agents + draws_sim[period, :])
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

        # Compute non-employment resources at current wages.
        non_emp_resources_agents = np.asarray(
            calculate_non_employment_consumption_resources(
                deductions_spec=model_spec.ssc_deductions,
                income_tax_spec=model_spec.tax_params,
                model_spec=model_spec,
                states=state_space[idx],
                log_wage_systematic=log_wage_agents[:, None],
                male_wage=male_wage,
                child_benefits=child_benefits,
                tax_splitting=model_spec.tax_splitting,
                hours=HOURS,
            )
        )[:, 0]

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
        current_states[:, IDX_EXP_STOCK] = _next_stock(
            stock=current_states[:, IDX_EXP_STOCK].astype(float),
            period=current_states[:, IDX_PERIOD].astype(int),
            init_exp_max=model_spec.init_exp_max,
            pt_increment=pt_increment,
            choice=choice,
        )

        # Bookkeeping for separate PT/FT experience years.
        current_states[:, IDX_EXP_PT] = np.where(
            choice == 1,
            current_states[:, IDX_EXP_PT] + 1,
            current_states[:, IDX_EXP_PT],
        )
        current_states[:, IDX_EXP_FT] = np.where(
            choice == 2,
            current_states[:, IDX_EXP_FT] + 1,
            current_states[:, IDX_EXP_FT],
        )

        current_states[:, IDX_PERIOD] = current_states[:, IDX_PERIOD] + 1
        current_states[:, IDX_LAGGED] = choice
        current_states[:, IDX_CHILD_AGE] = child_new_age
        current_states[:, IDX_PARTNER] = new_partner

    return data


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
