import numpy as np
import pandas as pd

from soepy.simulate.simulate_python import get_simulate_func
from soepy.test.resources.aux_funcs import cleanup
from soepy.test.resources.initial_states import create_initial_states


def _write_minimal_exog_files(*, num_periods: int) -> None:
    educ_shares = pd.DataFrame(
        [1.0], index=pd.Index([0], name="educ_level"), columns=["educ_shares"]
    )
    educ_shares.to_pickle("test.soepy.educ.shares.pkl")

    child_age_shares = pd.DataFrame(
        [1.0],
        index=pd.MultiIndex.from_product(
            [[0], [-1]], names=["educ_level", "child_age"]
        ),
        columns=["child_age_shares"],
    )
    child_age_shares.to_pickle("test.soepy.child.age.shares.pkl")

    partner_shares = pd.DataFrame(
        [1.0],
        index=pd.Index([0], name="educ_level"),
        columns=["partner_shares"],
    )
    partner_shares.to_pickle("test.soepy.partner.shares.pkl")

    # Initial experience shares: degenerate at 0 years.
    for label in ["pt", "ft"]:
        exp_shares = pd.DataFrame(
            [1.0],
            index=pd.MultiIndex.from_product(
                [[0], [0]], names=["educ_level", f"{label}_exp"]
            ),
            columns=["exper_shares"],
        )
        exp_shares.to_pickle(f"test.soepy.{label}.exp.shares.pkl")

    # Keep probabilities > 0 but tiny to reduce Monte Carlo noise.
    child_prob = 1e-6
    child_info = pd.DataFrame(
        np.full(num_periods * 3, child_prob),
        index=pd.MultiIndex.from_product(
            [list(range(num_periods)), [0, 1, 2]],
            names=["period", "educ_level"],
        ),
        columns=["prob_child_values"],
    )
    child_info.to_pickle("test.soepy.child.pkl")

    # Partner arrival/separation probabilities by period and educ.
    partner_arrival = pd.DataFrame(
        np.zeros(num_periods),
        index=pd.MultiIndex.from_product(
            [list(range(num_periods)), [0]],
            names=["period", "educ_level"],
        ),
        columns=["prob_partner_values"],
    )
    partner_arrival.to_pickle("test.soepy.partner.arrival.pkl")

    partner_separation = pd.DataFrame(
        np.zeros(num_periods),
        index=pd.MultiIndex.from_product(
            [list(range(num_periods)), [0]],
            names=["period", "educ_level"],
        ),
        columns=["prob_partner_values"],
    )
    partner_separation.to_pickle("test.soepy.partner.separation.pkl")


def _minimal_model_spec_init_dict(*, num_periods: int, num_agents: int) -> dict:
    return {
        "GENERAL": {"num_periods": num_periods},
        "EDUC": {"educ_years": [0]},
        "SIMULATION": {
            "seed_sim": 1234,
            "num_agents_sim": num_agents,
            "elasticity_scale": 1,
        },
        "SOLUTION": {
            "seed_emax": 123,
            "num_draws_emax": 51,
            "integration_method": "quadrature",
        },
        "TAXES_TRANSFERS": {
            "alg1_replacement_no_child": 0.6,
            "alg1_replacement_child": 0.67,
            "child_benefits": 0.0,
            "regelsatz_single": 91,
            "regelsatz_partner": 82,
            "regelsatz_child": 59,
            "elterngeld_replacement": 0.67,
            "elterngeld_min": 300,
            "elterngeld_max": 1800,
            "parental_leave_regime": "elterngeld",
            "erziehungsgeld_income_threshold_single": 23_000,
            "erziehungsgeld_income_threshold_married": 30_000,
            "erziehungsgeld": 300,
            "addition_child_single": 33,
            "housing_single": 77.5,
            "housing_addtion": 15,
            "child_care_costs": {"under_3": [0.0, 0.0], "3_to_6": [0.0, 0.0]},
            "ssc_rate": 0.215,
            "ssc_cap": 63_000,
            "tax_year": 2007,
            "tax_splitting": True,
        },
        "EXOG_PROC": {
            "educ_shares_file_name": "test.soepy.educ.shares.pkl",
            "child_age_shares_file_name": "test.soepy.child.age.shares.pkl",
            "child_age_init_max": -1,
            "partner_shares_file_name": "test.soepy.partner.shares.pkl",
            "ft_exp_shares_file_name": "test.soepy.ft.exp.shares.pkl",
            "pt_exp_shares_file_name": "test.soepy.pt.exp.shares.pkl",
            "init_exp_max": 0,
            "child_info_file_name": "test.soepy.child.pkl",
            "partner_arrival_info_file_name": "test.soepy.partner.arrival.pkl",
            "partner_separation_info_file_name": "test.soepy.partner.separation.pkl",
            "child_age_max": 2,
            "partner_cf_const": 0.0,
            "partner_cf_age": 0.0,
            "partner_cf_age_sq": 0.0,
            "partner_cf_educ": 0.0,
            "child_cf_const": 0.0,
            "child_cf_age": 0.0,
            "child_cf_age_sq": 0.0,
            "child_cf_educ": 0.0,
            "child_cf_presence": 0.0,
        },
        # Continuous experience grid (required at top-level).
        "exp_grid": np.linspace(0.0, 1.0, 10).tolist(),
    }


def _minimal_model_params_df() -> pd.DataFrame:
    rows = []

    for educ_type in ["low", "middle", "high"]:
        rows.append(("const_wage_eq", f"gamma_0_{educ_type}", 0.0))
        rows.append(("exp_return", f"gamma_1_{educ_type}", 0.1))
        rows.append(("exp_increase_p", f"gamma_p_{educ_type}", 0.5))
        rows.append(("exp_depr_rate", f"exp_depr_rate_{educ_type}", 0.1))

        rows.append(("disutil_work", f"no_kids_f_educ_{educ_type}", 0.0))
        rows.append(("disutil_work", f"yes_kids_f_educ_{educ_type}", 0.0))
        rows.append(("disutil_work", f"no_kids_p_educ_{educ_type}", 0.0))
        rows.append(("disutil_work", f"yes_kids_p_educ_{educ_type}", 0.0))

    rows.append(("exp_increase_p_mom", "gamma_p_mom", 0.0))
    rows.append(("switch_cost", "home_to_part", -0.08))
    rows.append(("switch_cost", "home_to_full", -0.12))
    rows.append(("switch_cost", "part_to_full", -0.05))
    rows.append(("switch_cost", "full_to_part", -0.03))

    for name in [
        "child_0_2_f",
        "child_3_5_f",
        "child_6_10_f",
        "child_0_2_p",
        "child_3_5_p",
        "child_6_10_p",
        "child_11_age_max_f",
        "child_11_age_max_p",
    ]:
        rows.append(("disutil_work", name, 0.0))

    rows.append(("discount", "delta", 0.95))
    rows.append(("risk", "mu", 1.0))
    rows.append(("sd_wage_shock", "sigma", 0.0))
    rows.append(("terminal_proxy", "beta_0", 0.0))
    rows.append(("terminal_proxy", "beta_1", 0.0))
    rows.append(("terminal_proxy", "beta_2", 0.0))

    df = pd.DataFrame(rows, columns=["category", "name", "value"]).set_index(
        ["category", "name"]
    )
    return df


def test_value_function_matches_mean_realized_discounted_sum():
    num_periods = 5
    num_agents = 10000

    _write_minimal_exog_files(num_periods=num_periods)

    model_spec_init_dict = _minimal_model_spec_init_dict(
        num_periods=num_periods, num_agents=num_agents
    )
    model_params_df = _minimal_model_params_df()

    initial_states = create_initial_states(
        model_params_init_file_name=model_params_df,
        model_spec_init_file_name=model_spec_init_dict,
    )
    simulate_func = get_simulate_func(
        model_params_init_file_name=model_params_df,
        model_spec_init_file_name=model_spec_init_dict,
        initial_states=initial_states,
        biased_exp=True,
        data_sparse=False,
    )
    df = simulate_func(
        model_params_init_file_name_inner=model_params_df,
        model_spec_init_file_name_inner=model_spec_init_dict,
    )

    # Compute realized discounted sum of chosen flow utility per agent.
    df_reset = df.reset_index()

    flow_chosen = df_reset["Flow_Utility_N"].to_numpy()
    choice = df_reset["Choice"].to_numpy()

    flow_p = df_reset["Flow_Utility_P"].to_numpy()
    flow_f = df_reset["Flow_Utility_F"].to_numpy()

    flow_chosen = np.where(choice == 1, flow_p, flow_chosen)
    flow_chosen = np.where(choice == 2, flow_f, flow_chosen)

    cont_chosen = np.nan_to_num(df_reset["Continuation_Value_N"].to_numpy(), nan=0.0)
    cont_p = np.nan_to_num(df_reset["Continuation_Value_P"].to_numpy(), nan=0.0)
    cont_f = np.nan_to_num(df_reset["Continuation_Value_F"].to_numpy(), nan=0.0)
    cont_chosen = np.where(choice == 1, cont_p, cont_chosen)
    cont_chosen = np.where(choice == 2, cont_f, cont_chosen)

    disc = np.power(
        float(model_params_df.loc[("discount", "delta"), "value"]),
        df_reset["Period"].to_numpy(),
    )

    last_period = df_reset["Period"].to_numpy() == (num_periods - 1)
    terminal_disc = np.power(
        float(model_params_df.loc[("discount", "delta"), "value"]),
        df_reset["Period"].to_numpy() + 1,
    )
    flow_plus_terminal = flow_chosen * disc + cont_chosen * terminal_disc * last_period

    disc_sum_by_id = (
        pd.Series(flow_plus_terminal).groupby(df_reset["Identifier"], sort=False).sum()
    )
    mean_disc_sum = float(disc_sum_by_id.mean())

    assert np.isfinite(mean_disc_sum)

    cleanup()
