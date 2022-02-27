from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import pmap
from jax import vmap
from jax.config import config

from soepy.shared.shared_constants import NUM_CHOICES
from soepy.shared.tax_and_transfers import calculate_net_income

config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(1,))
def get_max_aggregated_utilities_jax(
    delta,
    tax_splitting,
    mu,
    hours,
    deductions_spec,
    income_tax_spec,
    child_care_costs,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    cont_values,
    non_employment_consumption_resources,
    male_wage,
    child_benefits,
    equivalence,
    child_care_bin,
    partner_indicator,
):
    consumption_not_working = non_employment_consumption_resources / equivalence
    current_max_value_function = calc_value_func_from_cons(
        consumption_not_working, non_consumption_utilities[0], delta, mu, cont_values[0]
    )

    for j in range(1, NUM_CHOICES):
        female_wage = hours[j] * jnp.exp(log_wage_systematic + draws[j - 1])

        net_income = calculate_net_income(
            income_tax_spec,
            deductions_spec,
            female_wage,
            male_wage,
            partner_indicator,
            tax_splitting,
        )

        child_costs = child_care_costs[child_care_bin, j - 1]

        consumption = (
            jnp.maximum(net_income + child_benefits - child_costs, 1e-20) / equivalence
        )
        value_function_choice = calc_value_func_from_cons(
            consumption, non_consumption_utilities[j], delta, mu, cont_values[j]
        )

        current_max_value_function = jnp.maximum(
            value_function_choice, current_max_value_function
        )

    return current_max_value_function


@jit
def calc_value_func_from_cons(cons, non_cons_utility, delta, mu, emax):
    consumption_utility = (cons ** mu) / mu

    return consumption_utility * non_cons_utility + delta * emax


@jit
def weighting_emax(child_emaxs, prob_child, prob_partner):
    """
    This weights the child state emaxs according to the exogenous process probabilities.
    """
    weight_01 = (1 - prob_child) * prob_partner[1] * child_emaxs[0, 1]
    weight_00 = (1 - prob_child) * prob_partner[0] * child_emaxs[0, 0]
    weight_10 = prob_child * prob_partner[0] * child_emaxs[1, 0]
    weight_11 = prob_child * prob_partner[1] * child_emaxs[1, 1]
    return weight_11 + weight_10 + weight_00 + weight_01


@partial(jit, static_argnums=(2,))
def construct_emax_jax(
    delta,
    mu,
    tax_splitting,
    draws,
    hours,
    deductions_spec,
    income_tax_spec,
    child_care_costs,
    log_wage_systematic,
    non_consumption_utilities,
    continuation_values,
    non_employment_consumption_resources,
    index_child_care_costs,
    male_wage,
    child_benefits,
    equivalence,
    partner_indicator,
):
    # """Simulate expected maximum utility for a given distribution of the unobservables.
    #
    # The function calculates the maximum expected value function over the distribution of
    # the error term at each state space point in the period currently reached by the
    # parent loop. The expectation calculation is performed via `Monte Carlo
    # integration`. The goal is to approximate an integral by evaluating the integrand at
    # randomly chosen points. In this setting, one wants to approximate the expected
    # maximum utility of a given state.
    #
    # Parameters
    # ----------
    # delta : int
    #     Dynamic discount factor.
    # log_wage_systematic : array
    #     One dimensional array with length num_states containing the part of the wages
    #     at the respective state space point that do not depend on the agent's choice,
    #     nor on the random shock.
    # budget_constraint_components : array
    #     One dimensional array with length num_states containing monetary components
    #     that influence the budget available for consumption spending above and beyond
    #     own labor and non-labor income. Currently containing partner earnings
    #     in the case that a partner is present.
    # non_consumption_utilities : np.ndarray
    #     Array of dimension (num_states, num_choices) containing the utility
    #     contribution of non-pecuniary factors.
    # draws : np.ndarray
    #     Array of dimension (num_periods, num_choices, num_draws). Randomly drawn
    #     realisations of the error term used to integrate out the distribution of
    #     the error term.
    # emaxs : np.ndarray
    #     An array of dimension (num. states in period, num choices + 1).
    #     The object's rows contain the continuation values of each choice at the specific
    #     state space points as its first elements. The last row element corresponds
    #     to the maximum expected value function of the state. This column is
    #     full of zeros for the input object.
    # hours : np.array
    #     Array of constants, corresponding to the working hours associated with
    #     each employment choice.
    # mu : int
    #     Constant governing the degree of risk aversion and inter-temporal
    #     substitution in the model.
    # benefits : int
    #     Constant level of hourly income received in case of choice N,
    #     non-employment.
    #
    # Returns
    # -------
    # emax : np.array
    #     Expected maximum value function of the current state space point.
    #     Array of length number of states in the current period. The vector
    #     corresponds to the second block of values in the data:`emaxs` object.
    #
    # .. _Monte Carlo integration:
    #     https://en.wikipedia.org/wiki/Monte_Carlo_integration
    #
    # """

    partial_max_ut = partial(
        get_max_aggregated_utilities_jax,
        delta,
        tax_splitting,
        mu,
        hours,
        deductions_spec,
        income_tax_spec,
        child_care_costs,
    )
    emax_3 = vmap(
        vmap(
            partial_max_ut,
            in_axes=(None, None, 0, None, None, None, None, None, None, None,),
        ),
        in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0),
    )(
        log_wage_systematic,
        non_consumption_utilities,
        draws,
        continuation_values,
        non_employment_consumption_resources,
        male_wage,
        child_benefits,
        equivalence,
        index_child_care_costs,
        partner_indicator,
    ).mean(
        axis=1
    )

    return emax_3
