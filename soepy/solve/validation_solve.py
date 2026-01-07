import numba
import numpy as np

from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.tax_and_transfers import calculate_net_income


@numba.njit(nogil=True)
def _get_max_aggregated_utilities_validation(
    delta,
    baby_child,
    log_wage_systematic,
    non_consumption_utilities,
    draw,
    emaxs,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    male_wage,
    child_benefits,
    equivalence,
    tax_splitting,
    child_care_costs,
    child_care_bin,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
):

    consumption_0 = non_employment_consumption_resources / equivalence

    current_max_value_function = (consumption_0**mu / mu) * non_consumption_utilities[
        0
    ] + delta * emaxs[0]

    for j in range(1, 3):
        female_wage = hours[j] * np.exp(log_wage_systematic + draw)

        net_income = calculate_net_income(
            income_tax_spec, deductions_spec, female_wage, male_wage, tax_splitting
        )
        net_income += (
            (child_benefits > 0)
            * (j == 1)
            * calc_erziehungsgeld(
                male_wage,
                female_wage,
                male_wage > 0,
                baby_child,
                erziehungsgeld_inc_single,
                erziehungsgeld_inc_married,
                erziehungsgeld,
            )
        )

        child_costs = child_care_costs[child_care_bin, j - 1]

        consumption = (
            max(net_income + child_benefits - child_costs, 1e-14) / equivalence
        )

        consumption_utility = consumption**mu / mu

        value_function_choice = (
            consumption_utility * non_consumption_utilities[j] + delta * emaxs[j]
        )

        current_max_value_function = np.maximum(
            current_max_value_function, value_function_choice
        )

    return current_max_value_function


@numba.njit(nogil=True)
def do_weighting_emax(child_emaxs, prob_child, prob_partner):
    weight_01 = (1 - prob_child) * prob_partner[1] * child_emaxs[0, 1]
    weight_00 = (1 - prob_child) * prob_partner[0] * child_emaxs[0, 0]
    weight_10 = prob_child * prob_partner[0] * child_emaxs[1, 0]
    weight_11 = prob_child * prob_partner[1] * child_emaxs[1, 1]
    return weight_11 + weight_10 + weight_00 + weight_01


@numba.guvectorize(
    [
        "f8, b1, f8, f8[:], f8[:], f8[:], f8[:, :, :], f8, f8[:], f8[:], f8, f8, f8[:], "
        "f8[:, :], f8[:, :], i8, f8, f8, f8, f8, f8, f8, b1, f8[:], f8[:]"
    ],
    "(), (), (), (n_choices), (n_draws), (n_draws), (n_choices, n_children_states, "
    "n_partner_states), (), (n_partner_states), (n_choices), (), "
    "(), (n_ssc_params), (n_tax_params, n_tax_params), (n_choices, "
    "n_age_child_costs), (), (), (), (), (), (), (), (), (num_outputs) -> (num_outputs)",
    nopython=True,
    # target="cpu",
    target="parallel",
)
def construct_emax_validation(
    delta,
    baby_child,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    draw_weights,
    emaxs_child_states,
    prob_child,
    prob_partner,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    child_care_costs,
    index_child_care_costs,
    male_wage,
    child_benefits,
    equivalence,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
    tax_splitting,
    dummy_array,
    emax,
):
    """Simulate expected maximum utility for a given distribution of the unobservables.

    The function calculates the maximum expected value function over the distribution of
    the error term at each state space point in the period currently reached by the
    parent loop. The expectation calculation is performed via `Monte Carlo
    integration`. The goal is to approximate an integral by evaluating the integrand at
    randomly chosen points. In this setting, one wants to approximate the expected
    maximum utility of a given state.

    Parameters
    ----------
    delta : int
        Dynamic discount factor.
    log_wage_systematic : array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    budget_constraint_components : array
        One dimensional array with length num_states containing monetary components
        that influence the budget available for consumption spending above and beyond
        own labor and non-labor income. Currently containing partner earnings
        in the case that a partner is present.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.
    draws : np.ndarray
        Array of dimension (num_periods, num_choices, num_draws). Randomly drawn
        realisations of the error term used to integrate out the distribution of
        the error term.
    emaxs : np.ndarray
        An array of dimension (num. states in period, num choices + 1).
        The object's rows contain the continuation values of each choice at the specific
        state space points as its first elements. The last row element corresponds
        to the maximum expected value function of the state. This column is
        full of zeros for the input object.
    hours : np.array
        Array of constants, corresponding to the working hours associated with
        each employment choice.
    mu : int
        Constant governing the degree of risk aversion and inter-temporal
        substitution in the model.
    benefits : int
        Constant level of hourly income received in case of choice N,
        non-employment.

    Returns
    -------
    emax : np.array
        Expected maximum value function of the current state space point.
        Array of length number of states in the current period. The vector
        corresponds to the second block of values in the data:`emaxs` object.

    .. _Monte Carlo integration:
        https://en.wikipedia.org/wiki/Monte_Carlo_integration

    """
    num_draws = draws.shape[0]

    emax[0] = do_weighting_emax(emaxs_child_states[0, :, :], prob_child, prob_partner)
    emax[1] = do_weighting_emax(emaxs_child_states[1, :, :], prob_child, prob_partner)
    emax[2] = do_weighting_emax(emaxs_child_states[2, :, :], prob_child, prob_partner)

    emax[3] = 0.0

    for i in range(num_draws):
        max_total_utility = _get_max_aggregated_utilities_validation(
            delta,
            baby_child,
            log_wage_systematic,
            non_consumption_utilities,
            draws[i],
            emax[:3],
            hours,
            mu,
            non_employment_consumption_resources,
            deductions_spec,
            income_tax_spec,
            male_wage,
            child_benefits,
            equivalence,
            tax_splitting,
            child_care_costs,
            index_child_care_costs,
            erziehungsgeld_inc_single,
            erziehungsgeld_inc_married,
            erziehungsgeld,
        )

        emax[3] += max_total_utility * draw_weights[i]
