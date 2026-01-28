import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.experience_stock import get_pt_increment
from soepy.shared.experience_stock import next_stock
from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.wages import calculate_log_wage
from soepy.simulate.constants_sim import DATA_FORMATS_SIM
from soepy.simulate.constants_sim import DATA_FORMATS_SPARSE
from soepy.simulate.constants_sim import DATA_LABLES_SIM
from soepy.simulate.constants_sim import LABELS_DATA_SPARSE
from soepy.simulate.income_sim import calculate_employment_consumption_resources
from soepy.simulate.initial_states import prepare_simulation_data


def _get_state_col_positions():
    """Return column positions for the state vector stored in `current_states`.

    The simulation uses a NumPy array for speed. Column positions are derived from
    `DATA_LABLES_SIM` to avoid hard-coded IDX globals.
    """

    state_labels = DATA_LABLES_SIM[:10]
    return {label: i for i, label in enumerate(state_labels)}


def _get_sparse_state_positions(state_col_positions):
    labels_sparse_states = LABELS_DATA_SPARSE[:-2]
    return np.array(
        [state_col_positions[label] for label in labels_sparse_states], dtype=int
    )


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
    biased_exp,
    data_sparse=False,
):
    """Simulate agent histories under the continuous-experience model."""

    np.random.seed(model_spec.seed_sim)

    emaxs = np.asarray(emaxs)
    non_consumption_utilities = np.asarray(non_consumption_utilities)

    initial_states, draws_sim = prepare_simulation_data(
        model_params=model_params,
        model_spec=model_spec,
        prob_educ_level=prob_educ_level,
        prob_child_age=prob_child_age,
        prob_partner_present=prob_partner_present,
        prob_exp_pt=prob_exp_pt,
        prob_exp_ft=prob_exp_ft,
        biased_exp=biased_exp,
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
        biased_exp=biased_exp,
        data_sparse=data_sparse,
    )

    stacked = np.vstack(data)

    if data_sparse:
        # Ensure sparse state columns are integers (except identifier, wages).
        dataset = pd.DataFrame(stacked, columns=pd.Index(LABELS_DATA_SPARSE)).astype(
            DATA_FORMATS_SPARSE
        )
        labels = LABELS_DATA_SPARSE
    else:
        dataset = pd.DataFrame(stacked, columns=pd.Index(DATA_LABLES_SIM)).astype(
            DATA_FORMATS_SIM
        )
        labels = DATA_LABLES_SIM

    # Avoid string-based indexing on the DataFrame.
    choice_pos = labels.index("Choice")
    wage_pos = labels.index("Wage_Observed")
    choice_arr = dataset.iloc[:, choice_pos].to_numpy()
    dataset.iloc[choice_arr == 0, wage_pos] = np.nan

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
    biased_exp,
    data_sparse,
):
    state_col = _get_state_col_positions()
    state_labels = list(state_col)

    sparse_state_labels = LABELS_DATA_SPARSE[:-2]

    data = []
    current_states = initial_states[state_labels].iloc[0:0].copy()

    for period in range(model_spec.num_periods):
        entrants = initial_states.loc[initial_states.Period.eq(period), state_labels]
        current_states = pd.concat([current_states, entrants], ignore_index=True)

        age_child = current_states.iloc[:, state_col["Age_Youngest_Child"]].to_numpy()
        age_idx = np.where(
            age_child == -1,
            indexer.shape[4] - 1,
            age_child,
        )

        idx = indexer[
            current_states.iloc[:, state_col["Period"]].to_numpy(),
            current_states.iloc[:, state_col["Education_Level"]].to_numpy(),
            current_states.iloc[:, state_col["Lagged_Choice"]].to_numpy(),
            current_states.iloc[:, state_col["Type"]].to_numpy(),
            age_idx,
            current_states.iloc[:, state_col["Partner_Indicator"]].to_numpy(),
        ]

        stock = current_states.iloc[:, state_col["Experience_Stock"]].to_numpy(
            dtype=float
        )

        # Interpolate continuation values on the experience grid.
        continuation_grid = emaxs[idx, :, :NUM_CHOICES]
        continuation_grid = np.transpose(continuation_grid, (0, 2, 1))
        continuation_values = _interp_uniform_grid(continuation_grid, stock[:, None])

        non_cons_util_agents = non_consumption_utilities[idx]

        educ_level = current_states.iloc[:, state_col["Education_Level"]].to_numpy()
        pt_increment = get_pt_increment(
            model_params=model_params,
            educ_level=educ_level,
            child_age=age_child,
            biased_exp=biased_exp,
        )

        log_wage_agents = np.asarray(
            calculate_log_wage(
                model_params=model_params,
                educ=educ_level,
                exp_stock=stock,
                init_exp_max=model_spec.init_exp_max,
                pt_increment=pt_increment,
                period=period,
            )
        )

        wages = np.exp(log_wage_agents + draws_sim[period, : len(current_states)])
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
            married = (
                current_states.iloc[:, state_col["Partner_Indicator"]].to_numpy() == 1
            )
            baby_child = (age_child == 0) | (age_child == 1)

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

        current_states_np = current_states.to_numpy()
        if data_sparse:
            rows = np.column_stack(
                (
                    current_states.loc[:, sparse_state_labels].to_numpy(),
                    choice,
                    wages,
                )
            )
        else:
            rows = np.column_stack(
                (
                    current_states_np,
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
        child_current_age = age_child

        if period == model_spec.num_periods - 1:
            child_new_age = child_current_age
        elif period <= model_spec.last_child_bearing_period:
            kids_draw = np.random.binomial(
                size=len(current_states),
                n=1,
                p=prob_child[period + 1, educ_level],
            )
            child_new_age = np.where(kids_draw == 0, child_age_update_rule[idx], 0)
        else:
            child_new_age = child_age_update_rule[idx]

        current_partner = current_states.iloc[
            :, state_col["Partner_Indicator"]
        ].to_numpy()
        new_partner = current_partner.copy()

        no_partner = current_partner == 0
        if no_partner.any():
            arr = np.random.binomial(
                size=no_partner.sum(),
                n=1,
                p=prob_partner[period, educ_level[no_partner], 0, 1],
            )
            new_partner[no_partner] = arr

        has_partner = current_partner == 1
        if has_partner.any():
            sep = np.random.binomial(
                size=has_partner.sum(),
                n=1,
                p=prob_partner[period, educ_level[has_partner], 1, 0],
            )
            new_partner[has_partner] = current_partner[has_partner] - sep

        # --- endogenous updates
        stock_next = np.asarray(
            next_stock(
                stock=stock,
                period=current_states.iloc[:, state_col["Period"]].to_numpy(),
                init_exp_max=model_spec.init_exp_max,
                pt_increment=pt_increment,
                choice=choice,
            )
        )

        current_states.iloc[:, state_col["Experience_Stock"]] = stock_next

        current_states.iloc[:, state_col["Experience_Part_Time"]] = np.where(
            choice == 1,
            current_states.iloc[:, state_col["Experience_Part_Time"]] + 1,
            current_states.iloc[:, state_col["Experience_Part_Time"]],
        )
        current_states.iloc[:, state_col["Experience_Full_Time"]] = np.where(
            choice == 2,
            current_states.iloc[:, state_col["Experience_Full_Time"]] + 1,
            current_states.iloc[:, state_col["Experience_Full_Time"]],
        )

        current_states.iloc[:, state_col["Period"]] = (
            current_states.iloc[:, state_col["Period"]] + 1
        )
        current_states.iloc[:, state_col["Lagged_Choice"]] = choice
        current_states.iloc[:, state_col["Age_Youngest_Child"]] = child_new_age
        current_states.iloc[:, state_col["Partner_Indicator"]] = new_partner

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
