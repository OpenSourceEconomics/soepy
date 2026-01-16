import jax
import jax.numpy as jnp
import numpy as np

from soepy.shared.tax_and_transfers_jax import calculate_net_income


def _get_max_aggregated_utilities(
    delta,
    log_wage_systematic,
    non_consumption_utilities,
    draw,
    draw_weight,
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
):
    consumption_0 = non_employment_consumption_resources / equivalence

    current_max_value_function = (consumption_0**mu / mu) * non_consumption_utilities[
        0
    ] + delta * emaxs[0]

    for j in range(1, 3):
        female_wage = hours[j] * jnp.exp(log_wage_systematic + draw)

        net_income = calculate_net_income(
            income_tax_spec, deductions_spec, female_wage, male_wage, tax_splitting
        )

        child_costs = child_care_costs[child_care_bin, j - 1]

        consumption = (net_income + child_benefits - child_costs).clip(
            min=1e-14
        ) / equivalence

        consumption_utility = consumption**mu / mu

        value_function_choice = (
            consumption_utility * non_consumption_utilities[j] + delta * emaxs[j]
        )

        current_max_value_function = jnp.maximum(
            current_max_value_function, value_function_choice
        )
    return current_max_value_function * draw_weight


def do_weighting_emax_scalar(child_emaxs, prob_child, prob_partner):
    weight_01 = (1 - prob_child) * prob_partner[1] * child_emaxs[0, 1]
    weight_00 = (1 - prob_child) * prob_partner[0] * child_emaxs[0, 0]
    weight_10 = prob_child * prob_partner[0] * child_emaxs[1, 0]
    weight_11 = prob_child * prob_partner[1] * child_emaxs[1, 1]
    return weight_11 + weight_10 + weight_00 + weight_01


def emax_weighting(emaxs_child_states, prob_child, prob_partner):
    return jax.vmap(
        jax.vmap(do_weighting_emax_scalar, in_axes=(0, None, None)), in_axes=(0, 0, 0)
    )(emaxs_child_states, prob_child, prob_partner)


def construct_emax(
    delta,
    log_wages_systematic,
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
    male_wages,
    child_benefits,
    equivalence_scales,
    tax_splitting,
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

    emax = emax_weighting(emaxs_child_states, prob_child, prob_partner)

    def max_aggregated_utilities_broadcast(
        log_wage_systematic_choices,
        non_consumption_utilities_choices,
        emax_choices,
        non_employment_consumption_resources_choices,
        male_wage,
        child_benefit,
        equivalence,
        index_child_care_cost,
        draw,
        draw_weight,
    ):
        return _get_max_aggregated_utilities(
            delta=delta,
            log_wage_systematic=log_wage_systematic_choices,
            non_consumption_utilities=non_consumption_utilities_choices,
            draw=draw,
            draw_weight=draw_weight,
            emaxs=emax_choices,
            hours=hours,
            mu=mu,
            non_employment_consumption_resources=non_employment_consumption_resources_choices,
            deductions_spec=deductions_spec,
            income_tax_spec=income_tax_spec,
            male_wage=male_wage,
            child_benefits=child_benefit,
            equivalence=equivalence,
            tax_splitting=tax_splitting,
            child_care_costs=jnp.asarray(child_care_costs),
            child_care_bin=index_child_care_cost,
        )

    emaxs_current_states = jax.vmap(
        jax.vmap(
            max_aggregated_utilities_broadcast,
            in_axes=(None, None, None, None, None, None, None, None, 0, 0),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None),
    )(
        log_wages_systematic,
        non_consumption_utilities,
        emax,
        non_employment_consumption_resources,
        male_wages,
        child_benefits,
        equivalence_scales,
        index_child_care_costs,
        draws,
        draw_weights,
    )
    emax_expected = emaxs_current_states.sum(axis=1)
    # Add emax expected as last column
    emaxs_with_expected = jnp.concatenate([emax, emax_expected[:, None]], axis=1)
    return emaxs_with_expected
