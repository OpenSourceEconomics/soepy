import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.experience_stock import exp_years_to_stock
from soepy.shared.experience_stock import get_pt_increment
from soepy.shared.experience_stock import next_stock
from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.non_employment import calculate_non_employment_consumption_resources
from soepy.shared.numerical_integration import draw_zero_one_distributed_shocks
from soepy.shared.tax_and_transfers_jax import calculate_net_income
from soepy.shared.wages import calculate_log_wage
from soepy.simulate.constants_sim import DATA_FORMATS_SPARSE
from soepy.simulate.income_sim import calculate_employment_consumption_resources

JAX_SIM_OUTPUTS = [
    "Identifier",
    "Period",
    "Education_Level",
    "Lagged_Choice",
    "Experience_Part_Time",
    "Experience_Full_Time",
    "Experience_Stock",
    "Type",
    "Age_Youngest_Child",
    "Partner_Indicator",
    "Choice",
    "Wage_Observed",
    "Potential_Wage",
    "Wage_Shock",
    "Male_Wages",
    "Equivalence_Scale",
    "Non_Consumption_Utility_N",
    "Non_Consumption_Utility_P",
    "Non_Consumption_Utility_F",
    "Flow_Utility_N",
    "Flow_Utility_P",
    "Flow_Utility_F",
    "Continuation_Value_N",
    "Continuation_Value_P",
    "Continuation_Value_F",
    "Value_Function_N",
    "Value_Function_P",
    "Value_Function_F",
    "Consumption_Resources_N",
    "Consumption_Resources_P",
    "Consumption_Resources_F",
    "Taste_Shock_N",
    "Taste_Shock_P",
    "Taste_Shock_F",
]


def pyth_simulate(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_consumption_utilities,
    child_age_update_rule,
    prob_child,
    prob_partner,
    biased_exp,
    initial_states,
    data_sparse=False,
):
    """Simulate agent histories under the continuous-experience model."""

    np.random.seed(model_spec.seed_sim)

    emaxs = np.asarray(emaxs)
    non_consumption_utilities = np.asarray(non_consumption_utilities)

    num_agents_sim = len(initial_states["Identifier"])
    type_shares = np.array(model_params.type_shares, dtype=float)
    num_types = len(type_shares)
    types = np.random.choice(np.arange(num_types), size=num_agents_sim, p=type_shares)

    initial_states = initial_states.copy()
    initial_states["Type"] = types

    pt_increment = get_pt_increment(
        model_params=model_params,
        educ_level=initial_states["Education_Level"].to_numpy(),
        child_age=initial_states["Age_Youngest_Child"].to_numpy(),
        biased_exp=False,
    )
    total_years = (
        initial_states["Experience_Part_Time"].to_numpy() * pt_increment
        + initial_states["Experience_Full_Time"].to_numpy()
    )
    initial_states["Experience_Stock"] = exp_years_to_stock(
        exp_years=total_years,
        period=initial_states["Period"].to_numpy(),
        init_exp_max=model_spec.init_exp_max,
    ).astype(float)

    draws_sim = draw_zero_one_distributed_shocks(
        model_spec.seed_sim,
        model_spec.num_periods,
        num_agents_sim,
    )
    educ_initial = initial_states["Education_Level"].to_numpy(dtype=int)
    draws_sim = (
        draws_sim
        * np.asarray(model_params.shock_sd, dtype=float)[educ_initial][None, :]
    )

    lambda_taste = float(model_params.lambda_taste)
    if lambda_taste > 0:
        taste_rng = np.random.default_rng(model_spec.seed_sim)
        taste_shocks_sim = taste_rng.gumbel(
            loc=0.0,
            scale=lambda_taste,
            size=(model_spec.num_periods, num_agents_sim, NUM_CHOICES),
        )
    else:
        taste_shocks_sim = np.zeros(
            (model_spec.num_periods, num_agents_sim, NUM_CHOICES), dtype=float
        )

    child_uniforms_sim = np.random.uniform(
        size=(model_spec.num_periods, num_agents_sim)
    )
    partner_arrival_uniforms_sim = np.random.uniform(
        size=(model_spec.num_periods, num_agents_sim)
    )
    partner_separation_uniforms_sim = np.random.uniform(
        size=(model_spec.num_periods, num_agents_sim)
    )

    data_list = simulate_agents_over_periods(
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
        taste_shocks_sim=taste_shocks_sim,
        child_uniforms_sim=child_uniforms_sim,
        partner_arrival_uniforms_sim=partner_arrival_uniforms_sim,
        partner_separation_uniforms_sim=partner_separation_uniforms_sim,
        initial_states=initial_states,
        model_params=model_params,
        biased_exp=biased_exp,
        data_sparse=data_sparse,
    )

    data = pd.concat(data_list)

    if data_sparse:
        # Ensure sparse state columns are integers (except identifier, wages).
        data = data.astype(DATA_FORMATS_SPARSE)

    # Alter observed wage for unemployed to nans
    choice_arr = data["Choice"].to_numpy()
    data["Potential_Wage"] = data["Wage_Observed"].copy()
    data.loc[choice_arr == 0, "Wage_Observed"] = np.nan

    return data


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
    taste_shocks_sim,
    child_uniforms_sim,
    partner_arrival_uniforms_sim,
    partner_separation_uniforms_sim,
    initial_states,
    model_params,
    biased_exp,
    data_sparse,
):

    max_entry_year = np.max(model_spec.educ_years)
    data = []
    # Create empty DataFrame with same columns and dtypes as initial_states.
    current_states = initial_states.iloc[0:0].copy()

    state_col = {label: i for i, label in enumerate(current_states.columns)}

    for period in range(model_spec.num_periods):
        if period <= max_entry_year:
            entrants = initial_states.loc[initial_states.Period.eq(period), :]
            current_states = pd.concat([current_states, entrants], ignore_index=True)

        age_child = current_states.iloc[:, state_col["Age_Youngest_Child"]].to_numpy()
        age_idx = np.where(
            age_child == -1,
            indexer.shape[4] - 1,
            age_child,
        )

        educ_level = current_states.iloc[:, state_col["Education_Level"]].to_numpy()
        partner_indicator = current_states.iloc[
            :, state_col["Partner_Indicator"]
        ].to_numpy()

        idx = indexer[
            current_states.iloc[:, state_col["Period"]].to_numpy(),
            educ_level,
            current_states.iloc[:, state_col["Lagged_Choice"]].to_numpy(),
            current_states.iloc[:, state_col["Type"]].to_numpy(),
            age_idx,
            partner_indicator,
        ]

        stock = current_states.iloc[:, state_col["Experience_Stock"]].to_numpy()

        # Interpolate continuation values on the experience grid.
        continuation_grid = emaxs[idx, :, :NUM_CHOICES]
        continuation_grid = np.transpose(continuation_grid, (0, 2, 1))
        continuation_values = _interp_uniform_grid(continuation_grid, stock[:, None])

        non_cons_util_agents = non_consumption_utilities[idx]

        log_wage_agents = np.asarray(
            calculate_log_wage(
                model_params=model_params,
                educ=educ_level,
                exp_stock=stock,
                init_exp_max=model_spec.init_exp_max,
                period=period,
            )
        )

        identifiers = current_states.iloc[:, state_col["Identifier"]].to_numpy()
        wage_shocks = draws_sim[period, identifiers]
        taste_shocks_agents = taste_shocks_sim[period, identifiers, :]
        wages = np.exp(log_wage_agents + wage_shocks)
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
            married = partner_indicator == 1
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
        non_emp_resources_agents, male_net_inc = np.asarray(
            calculate_non_employment_consumption_resources(
                deductions_spec=model_spec.ssc_deductions,
                income_tax_spec=model_spec.tax_params,
                model_spec=model_spec,
                states=state_space[idx],
                log_wage_systematic=log_wage_agents,
                male_wage=male_wage,
                child_benefits=child_benefits,
                tax_splitting=model_spec.tax_splitting,
                hours=HOURS,
                debug=True,
            )
        )

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
        choice = np.argmax(value_functions + taste_shocks_agents, axis=1)

        if data_sparse:
            this_period_df = current_states.copy()
            this_period_df["Choice"] = choice
            this_period_df["Wage_Observed"] = wages
        else:
            this_period_df = current_states.copy()
            this_period_df["Choice"] = choice
            this_period_df["Wage_Observed"] = wages
            this_period_df["Male_Wages"] = male_wage
            this_period_df["Wage_Shock"] = wage_shocks
            this_period_df["Equivalence_Scale"] = equiv_scale
            for i, append in enumerate(["N", "P", "F"]):
                this_period_df[
                    f"Non_Consumption_Utility_{append}"
                ] = non_cons_util_agents[:, i]
                this_period_df[f"Flow_Utility_{append}"] = flow_utilities[:, i]
                this_period_df[f"Continuation_Value_{append}"] = continuation_values[
                    :, i
                ]
                this_period_df[f"Value_Function_{append}"] = value_functions[:, i]
                this_period_df[
                    f"Consumption_Resources_{append}"
                ] = consumption_resources[:, i]
                this_period_df[f"Taste_Shock_{append}"] = taste_shocks_agents[:, i]

        data.append(this_period_df)

        # --- exogenous updates
        child_current_age = age_child

        if period == model_spec.num_periods - 1:
            child_new_age = child_current_age
        else:
            if prob_child.ndim == 2:
                prob_child_period = prob_child[period + 1, educ_level]
            else:
                raise ValueError("Old")

            # else:
            #     last_dim = prob_child.shape[-1]
            #     if last_dim >= 5:
            #         child_state = np.where(
            #             age_child < 0,
            #             0,
            #             np.where(
            #                 age_child <= 2,
            #                 1,
            #                 np.where(
            #                     age_child <= 5,
            #                     2,
            #                     np.where(
            #                         age_child <= 10,
            #                         3,
            #                         4,
            #                     ),
            #                 ),
            #             ),
            #         )
            #         child_state = np.minimum(child_state, last_dim - 1)
            #         prob_child_period = prob_child[
            #             period + 1, educ_level, partner_indicator, child_state
            #         ]
            #     elif last_dim == 3:
            #         child_state = np.where(
            #             age_child < 0,
            #             0,
            #             np.where(age_child <= 5, 1, 2),
            #         )
            #         prob_child_period = prob_child[
            #             period + 1, educ_level, partner_indicator, child_state
            #         ]
            #     elif last_dim == 2:
            #         has_kids = np.where(
            #             (age_child >= 0) & (age_child <= 10), 1, 0
            #         )
            #         prob_child_period = prob_child[
            #             period + 1, educ_level, partner_indicator, has_kids
            #         ]
            #     else:
            #         has_prior_kid = (age_child != -1).astype(int)
            #         prob_child_period = prob_child[
            #             period + 1, educ_level, partner_indicator, has_prior_kid
            #         ]
            kids_draw = (
                child_uniforms_sim[period, identifiers] < prob_child_period
            ).astype(int)
            child_new_age = np.where(kids_draw == 0, child_age_update_rule[idx], 0)

        new_partner = partner_indicator.copy()

        no_partner = partner_indicator == 0
        if no_partner.any():
            arr = (
                partner_arrival_uniforms_sim[period, identifiers[no_partner]]
                < prob_partner[period, educ_level[no_partner], 0, 1]
            ).astype(int)
            new_partner[no_partner] = arr

        has_partner = partner_indicator == 1
        if has_partner.any():
            sep = (
                partner_separation_uniforms_sim[period, identifiers[has_partner]]
                < prob_partner[period, educ_level[has_partner], 1, 0]
            ).astype(int)
            new_partner[has_partner] = partner_indicator[has_partner] - sep

        # --- endogenous updates
        stock_next = np.asarray(
            next_stock(
                stock=stock,
                period=period,
                init_exp_max=model_spec.init_exp_max,
                choice=choice,
                model_params=model_params,
                educ_level=educ_level,
                child_age=age_child,
                biased_exp=biased_exp,
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
    """Interpolate along the last axis on a uniform [0,1] grid. IMPORTANT: Function works only for
    experience grids generated with np.linspace(0, 1, n_grid).

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

    # Age bin 0 is no child, 1 is child age 0-2, 2 is child age 3-5, 3 is child age 6+.
    # We set child care costs to zero for age bin 3 and above, as we only model child care costs up to age 5.
    child_bins = child_bins.copy()
    child_bins[child_bins > 2] = 0

    child_costs = np.zeros((child_bins.shape[0], 2))
    for choice in range(2):
        for age_bin in range(1, 3):
            child_costs[child_bins == age_bin, choice] = child_care_costs[
                age_bin, choice
            ]
    return child_costs


def draw_simulation_randomness(model_spec, num_agents_sim):
    """Draw fixed simulation shocks with NumPy for the JAX simulation path."""

    np.random.seed(model_spec.seed_sim)
    type_uniforms = np.random.random(size=num_agents_sim)

    wage_draws = draw_zero_one_distributed_shocks(
        model_spec.seed_sim,
        model_spec.num_periods,
        num_agents_sim,
    )

    taste_rng = np.random.default_rng(model_spec.seed_sim)
    taste_shocks_standard = taste_rng.gumbel(
        loc=0.0,
        scale=1.0,
        size=(model_spec.num_periods, num_agents_sim, NUM_CHOICES),
    )

    child_uniforms = np.random.uniform(size=(model_spec.num_periods, num_agents_sim))
    partner_arrival_uniforms = np.random.uniform(
        size=(model_spec.num_periods, num_agents_sim)
    )
    partner_separation_uniforms = np.random.uniform(
        size=(model_spec.num_periods, num_agents_sim)
    )

    return {
        "type_uniforms": type_uniforms,
        "wage_draws": wage_draws,
        "taste_shocks_standard": taste_shocks_standard,
        "child_uniforms": child_uniforms,
        "partner_arrival_uniforms": partner_arrival_uniforms,
        "partner_separation_uniforms": partner_separation_uniforms,
    }


def initial_states_to_jax_array(initial_states):
    """Convert validated initial states to the fixed-width JAX simulation input."""

    return jnp.asarray(
        initial_states[
            [
                "Identifier",
                "Period",
                "Education_Level",
                "Lagged_Choice",
                "Experience_Part_Time",
                "Experience_Full_Time",
                "Age_Youngest_Child",
                "Partner_Indicator",
            ]
        ].to_numpy(),
    )


def jax_simulate_core(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_consumption_utilities,
    child_age_update_rule,
    prob_child,
    prob_partner,
    initial_states_array,
    type_uniforms,
    wage_draws,
    taste_shocks_standard,
    child_uniforms,
    partner_arrival_uniforms,
    partner_separation_uniforms,
    biased_exp,
):
    """Simulate histories as JAX arrays using pre-drawn NumPy randomness.

    The returned ``data`` has shape ``(num_periods, num_agents, n_columns)`` and
    columns listed in ``JAX_SIM_OUTPUTS``. Rows with ``active == False`` are
    padding rows for agents not yet in the model at that period.
    """

    identifiers = initial_states_array[:, 0].astype(jnp.int32)
    entry_period = initial_states_array[:, 1].astype(jnp.int32)
    educ_initial = initial_states_array[:, 2].astype(jnp.int32)
    initial_child_age = initial_states_array[:, 6].astype(jnp.int32)

    type_cdf = jnp.cumsum(jnp.asarray(model_params.type_shares))
    types = jnp.sum(type_uniforms[:, None] > type_cdf[None, :-1], axis=1).astype(
        jnp.int32
    )

    pt_increment = get_pt_increment(
        model_params=model_params,
        educ_level=educ_initial,
        child_age=initial_child_age,
        biased_exp=False,
    )
    total_years = initial_states_array[:, 4] * pt_increment + initial_states_array[:, 5]
    initial_stock = exp_years_to_stock(
        exp_years=total_years,
        period=entry_period,
        init_exp_max=model_spec.init_exp_max,
    )

    current_states = jnp.column_stack(
        [
            initial_states_array[:, 1],
            initial_states_array[:, 2],
            initial_states_array[:, 3],
            types.astype(float),
            initial_states_array[:, 6],
            initial_states_array[:, 7],
            initial_states_array[:, 4],
            initial_states_array[:, 5],
            initial_stock,
        ]
    )
    active = jnp.zeros(initial_states_array.shape[0], dtype=bool)

    carry = (current_states, active)
    scan_inputs = jnp.arange(model_spec.num_periods, dtype=jnp.int32)

    def scan_step(carry, period):
        current_states, active_prev = carry
        active_now = active_prev | (entry_period == period)

        period_output, current_states_next = jax_simulate_period(
            model_params=model_params,
            model_spec=model_spec,
            states=states,
            indexer=indexer,
            emaxs=emaxs,
            covariates=covariates,
            non_consumption_utilities=non_consumption_utilities,
            child_age_update_rule=child_age_update_rule,
            prob_child=prob_child,
            prob_partner=prob_partner,
            current_states=current_states,
            identifiers=identifiers,
            educ_initial=educ_initial,
            active=active_now,
            period=period,
            wage_draws=wage_draws,
            taste_shocks_standard=taste_shocks_standard,
            child_uniforms=child_uniforms,
            partner_arrival_uniforms=partner_arrival_uniforms,
            partner_separation_uniforms=partner_separation_uniforms,
            biased_exp=biased_exp,
        )

        return (current_states_next, active_now), (period_output, active_now)

    _, (data, active_mask) = jax.lax.scan(scan_step, carry, scan_inputs)
    return {"data": data, "active": active_mask}


def jax_simulate_period(
    model_params,
    model_spec,
    states,
    indexer,
    emaxs,
    covariates,
    non_consumption_utilities,
    child_age_update_rule,
    prob_child,
    prob_partner,
    current_states,
    identifiers,
    educ_initial,
    active,
    period,
    wage_draws,
    taste_shocks_standard,
    child_uniforms,
    partner_arrival_uniforms,
    partner_separation_uniforms,
    biased_exp,
):
    current_period = current_states[:, 0].astype(jnp.int32)
    educ_level = current_states[:, 1].astype(jnp.int32)
    lagged_choice = current_states[:, 2].astype(jnp.int32)
    unobs_type = current_states[:, 3].astype(jnp.int32)
    age_child = current_states[:, 4].astype(jnp.int32)
    partner_indicator = current_states[:, 5].astype(jnp.int32)
    exp_part_time = current_states[:, 6]
    exp_full_time = current_states[:, 7]
    stock = current_states[:, 8]

    age_idx = jnp.where(age_child == -1, indexer.shape[4] - 1, age_child)
    state_idx = indexer[
        current_period,
        educ_level,
        lagged_choice,
        unobs_type,
        age_idx,
        partner_indicator,
    ]

    continuation_grid = emaxs[state_idx, :, :NUM_CHOICES]
    continuation_grid = jnp.transpose(continuation_grid, (0, 2, 1))
    continuation_values = _interp_uniform_grid_jax(continuation_grid, stock[:, None])

    non_cons_util_agents = non_consumption_utilities[state_idx]

    log_wage_agents = calculate_log_wage(
        model_params=model_params,
        educ=educ_level,
        exp_stock=stock,
        init_exp_max=model_spec.init_exp_max,
        period=period,
    )

    wage_shocks = wage_draws[period, identifiers] * model_params.shock_sd[educ_initial]
    taste_shocks_agents = (
        taste_shocks_standard[period, identifiers, :] * model_params.lambda_taste
    )
    wages = jnp.exp(log_wage_agents + wage_shocks) * float(model_spec.elasticity_scale)

    female_income = wages[:, None] * jnp.asarray(HOURS[1:])[None, :]
    covariates_agents = covariates[state_idx]
    male_wage = covariates_agents[:, 1]
    child_benefits = covariates_agents[:, 3]
    equiv_scale = covariates_agents[:, 2]

    employment_resources = _jax_employment_consumption_resources(
        model_spec=model_spec,
        female_income=female_income,
        male_wage=male_wage,
    )

    if model_spec.parental_leave_regime == "erziehungsgeld":
        married = partner_indicator == 1
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
        employment_resources = employment_resources.at[:, 0].add(erz)

    employment_resources = employment_resources + child_benefits[:, None]
    employment_resources = employment_resources - _get_child_care_cost_for_choice_jax(
        covariates_agents[:, 0], model_spec.child_care_costs
    )

    non_emp_resources_agents, _ = calculate_non_employment_consumption_resources(
        deductions_spec=model_spec.ssc_deductions,
        income_tax_spec=model_spec.tax_params,
        model_spec=model_spec,
        states=states[state_idx],
        log_wage_systematic=log_wage_agents,
        male_wage=male_wage,
        child_benefits=child_benefits,
        tax_splitting=model_spec.tax_splitting,
        hours=jnp.asarray(HOURS),
        debug=True,
    )

    consumption_resources = jnp.column_stack(
        [non_emp_resources_agents, employment_resources]
    )
    consumption_resources = jnp.clip(consumption_resources, min=jnp.finfo(float).eps)

    flow_utilities = (
        (consumption_resources / equiv_scale[:, None]) ** model_params.mu
        / model_params.mu
        * non_cons_util_agents
    )
    value_functions = flow_utilities + model_params.delta * continuation_values
    choice = jnp.argmax(value_functions + taste_shocks_agents, axis=1).astype(jnp.int32)
    choice = jnp.where(active, choice, 0)

    child_new_age = _jax_next_child_age(
        model_spec=model_spec,
        child_age_update_rule=child_age_update_rule,
        prob_child=prob_child,
        child_uniforms=child_uniforms,
        state_idx=state_idx,
        identifiers=identifiers,
        period=period,
        educ_level=educ_level,
        age_child=age_child,
    )
    new_partner = _jax_next_partner(
        prob_partner=prob_partner,
        partner_arrival_uniforms=partner_arrival_uniforms,
        partner_separation_uniforms=partner_separation_uniforms,
        identifiers=identifiers,
        period=period,
        educ_level=educ_level,
        partner_indicator=partner_indicator,
    )

    stock_next = next_stock(
        stock=stock,
        period=period,
        init_exp_max=model_spec.init_exp_max,
        choice=choice,
        model_params=model_params,
        educ_level=educ_level,
        child_age=age_child,
        biased_exp=biased_exp,
    )

    next_states_candidate = jnp.column_stack(
        [
            current_period + 1,
            educ_level,
            choice,
            unobs_type,
            child_new_age,
            new_partner,
            exp_part_time + (choice == 1),
            exp_full_time + (choice == 2),
            stock_next,
        ]
    )
    current_states_next = jnp.where(
        active[:, None], next_states_candidate, current_states
    )

    observed_wage = jnp.where(choice == 0, jnp.nan, wages)
    period_output = jnp.column_stack(
        [
            identifiers,
            current_period,
            educ_level,
            lagged_choice,
            exp_part_time,
            exp_full_time,
            stock,
            unobs_type,
            age_child,
            partner_indicator,
            choice,
            observed_wage,
            wages,
            wage_shocks,
            male_wage,
            equiv_scale,
            non_cons_util_agents[:, 0],
            non_cons_util_agents[:, 1],
            non_cons_util_agents[:, 2],
            flow_utilities[:, 0],
            flow_utilities[:, 1],
            flow_utilities[:, 2],
            continuation_values[:, 0],
            continuation_values[:, 1],
            continuation_values[:, 2],
            value_functions[:, 0],
            value_functions[:, 1],
            value_functions[:, 2],
            consumption_resources[:, 0],
            consumption_resources[:, 1],
            consumption_resources[:, 2],
            taste_shocks_agents[:, 0],
            taste_shocks_agents[:, 1],
            taste_shocks_agents[:, 2],
        ]
    )
    period_output = jnp.where(active[:, None], period_output, 0.0)

    return period_output, current_states_next


def _jax_employment_consumption_resources(model_spec, female_income, male_wage):
    net_income = jax.vmap(calculate_net_income, in_axes=(None, None, 1, None, None),)(
        model_spec.tax_params,
        model_spec.ssc_deductions,
        female_income,
        male_wage,
        model_spec.tax_splitting,
    )
    return jnp.transpose(net_income)


def _jax_next_child_age(
    model_spec,
    child_age_update_rule,
    prob_child,
    child_uniforms,
    state_idx,
    identifiers,
    period,
    educ_level,
    age_child,
):
    if prob_child.ndim != 2:
        raise ValueError("Old")

    prob_child_period = prob_child[
        jnp.minimum(period + 1, model_spec.num_periods - 1), educ_level
    ]
    kids_draw = (child_uniforms[period, identifiers] < prob_child_period).astype(
        jnp.int32
    )
    next_age = jnp.where(kids_draw == 0, child_age_update_rule[state_idx], 0)
    return jnp.where(period == model_spec.num_periods - 1, age_child, next_age)


def _jax_next_partner(
    prob_partner,
    partner_arrival_uniforms,
    partner_separation_uniforms,
    identifiers,
    period,
    educ_level,
    partner_indicator,
):
    arrival = (
        partner_arrival_uniforms[period, identifiers]
        < prob_partner[period, educ_level, 0, 1]
    ).astype(jnp.int32)
    separation = (
        partner_separation_uniforms[period, identifiers]
        < prob_partner[period, educ_level, 1, 0]
    ).astype(jnp.int32)

    return jnp.where(partner_indicator == 0, arrival, partner_indicator - separation)


def _get_child_care_cost_for_choice_jax(child_bins, child_care_costs):
    child_bins = jnp.where(child_bins > 2, 0, child_bins).astype(jnp.int32)
    part_cost = jnp.where(
        child_bins == 1,
        child_care_costs[1, 0],
        jnp.where(child_bins == 2, child_care_costs[2, 0], 0.0),
    )
    full_cost = jnp.where(
        child_bins == 1,
        child_care_costs[1, 1],
        jnp.where(child_bins == 2, child_care_costs[2, 1], 0.0),
    )
    return jnp.column_stack([part_cost, full_cost])


def _interp_uniform_grid_jax(values, stock):
    n_grid = values.shape[-1]
    u = jnp.clip(stock, 0.0, 1.0) * (n_grid - 1)
    idx_lo = jnp.floor(u).astype(jnp.int32)
    idx_hi = jnp.minimum(idx_lo + 1, n_grid - 1)
    w = u - idx_lo
    take_lo = jnp.take_along_axis(values, idx_lo[..., None], axis=-1)[..., 0]
    take_hi = jnp.take_along_axis(values, idx_hi[..., None], axis=-1)[..., 0]
    return take_lo + w * (take_hi - take_lo)
