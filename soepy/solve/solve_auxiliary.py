import numpy as np
import numba
import pandas as pd

from soepy.shared.shared_constants import (
    MISSING_INT,
    NUM_CHOICES,
    INVALID_FLOAT,
    HOURS,
)
from soepy.shared.shared_auxiliary import calculate_deductions
from soepy.shared.shared_auxiliary import calculate_tax


@numba.jit(nopython=True)
def pyth_create_state_space(model_spec):
    """Create state space object.

    The state space consists of all admissible combinations of the following components:
    period, years of education, lagged choice, full-time experience (F),
    and part-time experience (P).

    :data:`states` stores the information on states in a tabular format.
    Each row of the table corresponds to one admissible state space point
    and contains the values of the state space components listed above.
    :data:`indexer` is a multidimensional array where each component
    of the state space corresponds to one dimension. The values of the array cells
    index the corresponding state space point in :data:`states`.
    Traversing the state space requires incrementing the indices of :data:`indexer`
    and selecting the corresponding state space point component values in :data:`states`.

    Parameters
    ----------
    model_spec: namedtuple
        Namedtuple containing all fixed parameters describing the model and its
         state space that are relevant for running a simulation.

    Returns
    -------
    states : np.ndarray
        Array with shape (num_states, 8) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment, type, age of the youngest child,
        indicator for the presence of a partner.
    indexer : np.ndarray
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.
    """
    data = []
    kids_ages = np.arange(-1, model_spec.child_age_max + 1)

    # Array for mapping the state space points (states) to indices
    shape = (
        model_spec.num_periods,
        model_spec.num_educ_levels,
        NUM_CHOICES,
        model_spec.num_periods,
        model_spec.num_periods,
        model_spec.num_types,
        kids_ages.shape[0],
        2,
    )

    indexer = np.full(shape, MISSING_INT)

    # Initialize counter for admissible state space points
    i = 0

    # Loop over all periods / all ages
    for period in range(model_spec.num_periods):

        # Loop over all types
        for type_ in range(model_spec.num_types):

            for partner_indicator in range(2):

                # Loop over all kids ages that are recorded
                for age_kid in kids_ages:
                    # Assumption: 1st kid is born no earlier than in period zero,
                    # i.e., in the current setup, no earlier than age 17.
                    # Can be relaxed, e.g., we assume that 1st kid can arrive earliest when
                    # a woman is 16 years old, the condition becomes:
                    # if age_kid > period + 1.
                    if age_kid - model_spec.child_age_init_max > period:
                        continue
                    # Make sure that women above 42 do not get kids
                    # For periods corresponding to ages > 40, the `age_kid`
                    # state space component can only take values -1, for no child ever,
                    # 11, for a child above 11, and 0 - 10 in such a fashion that no
                    # birth after 40 years of age is possible.
                    if (
                        period > model_spec.last_child_bearing_period
                        and 0
                        <= age_kid
                        <= min(period - (model_spec.last_child_bearing_period + 1), 10)
                    ):
                        continue

                    # Loop over all possible initial conditions for education
                    for educ_level in range(model_spec.num_educ_levels):

                        # Check if individual has already completed education
                        # and will make a labor supply choice in the period
                        if model_spec.educ_years[educ_level] > period:
                            continue

                        # Loop over all admissible years of experience
                        # accumulated in full-time
                        for exp_f in range(model_spec.num_periods):

                            # Loop over all admissible years of experience accumulated
                            # in part-time
                            for exp_p in range(model_spec.num_periods):

                                # The accumulation of experience cannot exceed time elapsed
                                # since individual entered the model
                                if (
                                    exp_f + exp_p
                                    > period - model_spec.educ_years[educ_level]
                                ):
                                    continue

                                # Add an additional entry state
                                # [educ_years + model_params.educ_min, 0, 0, 0]
                                # for individuals who have just completed education
                                # and still have no experience in any occupation.
                                if period == model_spec.educ_years[educ_level]:

                                    # Assign an additional integer count i
                                    # for entry state
                                    indexer[
                                        period,
                                        educ_level,
                                        0,
                                        0,
                                        0,
                                        type_,
                                        age_kid,
                                        partner_indicator,
                                    ] = i

                                    # Record the values of the state space components
                                    # for the currently reached entry state
                                    row = [
                                        period,
                                        educ_level,
                                        0,
                                        0,
                                        0,
                                        type_,
                                        age_kid,
                                        partner_indicator,
                                    ]

                                    # Update count once more
                                    i += 1

                                    data.append(row)

                                else:

                                    # Loop over the three labor market choices, N, P, F
                                    for choice_lagged in range(NUM_CHOICES):

                                        # If individual has only worked full-time in the past,
                                        # she can only have full-time (2) as lagged choice
                                        if (choice_lagged != 2) and (
                                            exp_f
                                            == period
                                            - model_spec.educ_years[educ_level]
                                        ):
                                            continue

                                        # If individual has only worked part-time in the past,
                                        # she can only have part-time (1) as lagged choice
                                        if (choice_lagged != 1) and (
                                            exp_p
                                            == period
                                            - model_spec.educ_years[educ_level]
                                        ):
                                            continue

                                        # If an individual has never worked full-time,
                                        # she cannot have that lagged activity
                                        if (choice_lagged == 2) and (exp_f == 0):
                                            continue

                                        # If an individual has never worked part-time,
                                        # she cannot have that lagged activity
                                        if (choice_lagged == 1) and (exp_p == 0):
                                            continue

                                        # If an individual has always been employed,
                                        # she cannot have non-employment (0) as lagged choice
                                        if (choice_lagged == 0) and (
                                            exp_f + exp_p
                                            == period
                                            - model_spec.educ_years[educ_level]
                                        ):
                                            continue

                                        # Check for duplicate states
                                        if (
                                            indexer[
                                                period,
                                                educ_level,
                                                choice_lagged,
                                                exp_p,
                                                exp_f,
                                                type_,
                                                age_kid,
                                                partner_indicator,
                                            ]
                                            != MISSING_INT
                                        ):
                                            continue

                                        # Assign the integer count i as an indicator for the
                                        # currently reached admissible state space point
                                        indexer[
                                            period,
                                            educ_level,
                                            choice_lagged,
                                            exp_p,
                                            exp_f,
                                            type_,
                                            age_kid,
                                            partner_indicator,
                                        ] = i

                                        # Update count
                                        i += 1

                                        # Record the values of the state space components
                                        # for the currently reached admissible
                                        # state space point
                                        row = [
                                            period,
                                            educ_level,
                                            choice_lagged,
                                            exp_p,
                                            exp_f,
                                            type_,
                                            age_kid,
                                            partner_indicator,
                                        ]

                                        data.append(row)

        states = np.array(data)

    # Return function output
    return states, indexer


def construct_covariates(states, model_spec):
    """Construct a matrix of all the covariates
    that depend only on the state space.

    Parameters
    ---------
    states : np.ndarray
        Array with shape (num_states, 8) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment, type, age of the youngest child,
        indicator for the presence of a partner.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, number of covariates) containing all additional
        covariates, which depend only on the state space information.

    """

    # Age youngest child
    # Bins of age of youngest child based on kids age
    # bin 0 corresponds to no kid, remaining bins as in Blundell
    # 0-2, 3-5, 6-10, 11+
    age_kid = pd.Series(states[:, 6])
    bins = pd.cut(
        age_kid,
        bins=[-2, -1, 2, 5, 10, 11],
        labels=[0, 1, 2, 3, 4],
    ).to_numpy()

    # Male wages based on age and education level of the woman
    # Wages are first calculated as hourly wages
    log_wages = (
        model_spec.partner_cf_const
        + model_spec.partner_cf_age * states[:, 0]
        + model_spec.partner_cf_age_sq * states[:, 0] ** 2
        + model_spec.partner_cf_educ * states[:, 1]
    )

    # Male wages
    # Final input of male wages / partner income is calculated on a weekly
    # basis. Underlying assumption that all men work full time.
    male_wages = np.where(states[:, 7] == 1, np.exp(log_wages) * HOURS[2], 0)

    # Equivalence scale
    # Depending on the presence of a partner and a child each state is
    # assigned an equivalence scale value following the modernized OECD
    # scale: 1 for a single woman HH, 1.5 for a woman with a partner,
    # 1.8 for a woman with a partner and a child and 1.3 for a woman with
    # a child and no partner
    equivalence_scale = np.full(states.shape[0], np.nan)
    equivalence_scale = np.where(
        (states[:, 6] == -1) & (states[:, 7] == 0), 1.0, equivalence_scale
    )
    equivalence_scale = np.where(
        (states[:, 6] == -1) & (states[:, 7] == 1), 1.5, equivalence_scale
    )
    equivalence_scale = np.where(
        (states[:, 6] != -1) & (states[:, 7] == 1), 1.8, equivalence_scale
    )
    equivalence_scale = np.where(
        (states[:, 6] != -1) & (states[:, 7] == 0), 1.3, equivalence_scale
    )

    assert (
        np.isnan(equivalence_scale).any() == 0
    ), "Some HH were not assigned an equivalence scale"

    # Child benefits
    # If a woman has a child she receives child benefits
    child_benefits = np.where(states[:, 6] == -1, 0, model_spec.child_benefits)

    # Collect in covariates vector
    covariates = np.column_stack((bins, male_wages, equivalence_scale, child_benefits))

    return covariates


def pyth_backward_induction(
    model_spec,
    states,
    indexer,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    covariates,
    child_age_update_rule,
    prob_child,
    prob_partner_arrival,
    prob_partner_separation,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
):
    """Get expected maximum value function at every state space point.
    Backward induction is performed all at once for all states in a given period.
    The function loops through each period. The included construct_emax function
    implicitly loops through all states in the period currently reached by the
    parent loop.

    Parameters
    ----------
    model_spec : namedtuple
        Contains all fixed parameters of the model including information on dimensions
        such as number of periods, agents, random draws, etc.
    states : np.ndarray
        Array with shape (num_states, 5) containing period, years of schooling,
        the lagged choice, the years of experience in part-time, and the
        years of experience in full-time employment.
    indexer : np.ndarray
        Array where each dimension represents a componenet of the state space.
        :data:`states[k]` returns the values of the state space components
        at state :data:`k`. Indexing :data:`indexer` by the same state space
        component values returns :data:`k`.
    log_wage_systematic : np.array
        One dimensional array with length num_states containing the part of the wages
        at the respective state space point that do not depend on the agent's choice,
        nor on the random shock.
    non_consumption_utilities : np.ndarray
        Array of dimension (num_states, num_choices) containing the utility
        contribution of non-pecuniary factors.

    Returns
    -------
    emaxs : np.ndarray
        An array of dimension (num_states, num choices + 1). The object's rows contain
        the continuation values of each choice at the specific state space points
        as its first elements. The last row element corresponds to the maximum
        expected value function of the state.
    """

    emaxs = np.zeros((states.shape[0], NUM_CHOICES + 1))

    # Loop backwards over all periods
    for period in reversed(range(model_spec.num_periods)):

        # Extract period information
        # States
        states_period = states[np.where(states[:, 0] == period)]

        # Info on updated age of child
        child_age_update_rule_period = child_age_update_rule[
            np.where(states[:, 0] == period)
        ]

        # Probability that a child arrives
        prob_child_period = prob_child[period]

        # Probability that a partner arrives
        prob_partner_arrival_period = prob_partner_arrival[period]
        prob_partner_separation_period = prob_partner_separation[period]

        # Period rewards
        log_wage_systematic_period = log_wage_systematic[states[:, 0] == period]
        non_consumption_utilities_period = non_consumption_utilities[
            states[:, 0] == period
        ]
        non_employment_consumption_resources_period = (
            non_employment_consumption_resources[states[:, 0] == period]
        )

        # Corresponding equivalence scale for period states
        male_wage_period = covariates[np.where(states[:, 0] == period)][:, 1]
        equivalence_scale_period = covariates[np.where(states[:, 0] == period)][:, 2]
        child_benefits_period = covariates[np.where(states[:, 0] == period)][:, 3]

        # Continuation value calculation not performed for last period
        # since continuation values are known to be zero
        if period == model_spec.num_periods - 1:
            pass
        else:

            # Fill first block of elements in emaxs for the current period
            # corresponding to the continuation values
            emaxs = get_continuation_values(
                model_spec,
                states_period,
                indexer,
                emaxs,
                child_age_update_rule_period,
                prob_child_period,
                prob_partner_arrival_period,
                prob_partner_separation_period,
            )

        # Extract current period information for current loop calculation
        emaxs_period = emaxs[np.where(states[:, 0] == period)]

        # Calculate emax for current period reached by the loop
        emax_period = construct_emax(
            model_spec.delta,
            log_wage_systematic_period,
            non_consumption_utilities_period,
            draws[period],
            emaxs_period[:, :3],
            HOURS,
            model_spec.mu,
            non_employment_consumption_resources_period,
            deductions_spec,
            income_tax_spec,
            male_wage_period,
            child_benefits_period,
            equivalence_scale_period,
        )
        emaxs_period[:, 3] = emax_period
        emaxs[np.where(states[:, 0] == period)] = emaxs_period

    return emaxs


@numba.njit(nogil=True)
def get_continuation_values(
    model_spec,
    states_subset,
    indexer,
    emaxs,
    child_age_update_rule_period,
    prob_child_period,
    prob_partner_arrival_period,
    prob_partner_separation_period,
):
    """Obtain continuation values for each of the choices at each state
    of the period currently reached by the parent loop.

    This function takes a parent node and looks up the continuation values
    of each of the available choices. It takes the entire block of
    data:`emaxs` corresponding to the period and fills in the first block
    of elements corresponding to the continuation values.
    The continuation value of each choice is the expected maximum value
    function of the next period's state if the particular choice was
    taken this period. The expected maximum value function values are
    contained as the last element of the data:`emaxs` row of next
    period's state.

    Warning
    -------
    This function must be extremely performant as the lookup is done for each state in a
    state space (except for states in the last period) for each evaluation of the
    optimization of parameters.
    """
    for i in range(states_subset.shape[0]):

        # Unpack parent state and get index
        (
            period,
            educ_level,
            choice_lagged,
            exp_p,
            exp_f,
            type_,
            age_kid,
            partner_indicator,
        ) = states_subset[i]

        k_parent = indexer[
            period,
            educ_level,
            choice_lagged,
            exp_p,
            exp_f,
            type_,
            age_kid,
            partner_indicator,
        ]

        # Child: No arrival
        # Choice: Non-employment
        k_0_00 = indexer[
            period + 1,
            educ_level,
            0,
            exp_p,
            exp_f,
            type_,
            child_age_update_rule_period[i],
            0,  # No partner
        ]

        k_0_01 = indexer[
            period + 1,
            educ_level,
            0,
            exp_p,
            exp_f,
            type_,
            child_age_update_rule_period[i],
            1,  # Partner
        ]

        # Choice: Part-time
        k_1_00 = indexer[
            period + 1,
            educ_level,
            1,
            exp_p + 1,
            exp_f,
            type_,
            child_age_update_rule_period[i],
            0,  # No partner
        ]

        k_1_01 = indexer[
            period + 1,
            educ_level,
            1,
            exp_p + 1,
            exp_f,
            type_,
            child_age_update_rule_period[i],
            1,  # Partner
        ]

        # Choice: Full-time
        k_2_00 = indexer[
            period + 1,
            educ_level,
            2,
            exp_p,
            exp_f + 1,
            type_,
            child_age_update_rule_period[i],
            0,  # No partner
        ]

        k_2_01 = indexer[
            period + 1,
            educ_level,
            2,
            exp_p,
            exp_f + 1,
            type_,
            child_age_update_rule_period[i],
            1,  # Partner
        ]

        # Child possible, integrate out partner and child probability
        if period <= model_spec.last_child_bearing_period:

            # Child arrives
            # Choice: Non-employment
            k_0_10 = indexer[
                period + 1,
                educ_level,
                0,
                exp_p,
                exp_f,
                type_,
                0,
                0,  # No partner
            ]

            k_0_11 = indexer[
                period + 1,
                educ_level,
                0,
                exp_p,
                exp_f,
                type_,
                0,
                1,  # Partner
            ]

            # Choice: Part-time
            k_1_10 = indexer[
                period + 1,
                educ_level,
                1,
                exp_p + 1,
                exp_f,
                type_,
                0,
                0,  # No partner
            ]

            k_1_11 = indexer[
                period + 1,
                educ_level,
                1,
                exp_p + 1,
                exp_f,
                type_,
                0,
                1,  # Partner
            ]

            # Choice: Full-time
            k_2_10 = indexer[
                period + 1,
                educ_level,
                2,
                exp_p,
                exp_f + 1,
                type_,
                0,
                0,  # No partner
            ]

            k_2_11 = indexer[
                period + 1,
                educ_level,
                2,
                exp_p,
                exp_f + 1,
                type_,
                0,
                1,  # Partner
            ]

            # Calculate E-Max
            if partner_indicator == 1:
                # Partner is present in the parent (current) state, i.e.,
                # partner remains or is lost in the child (future) state
                emaxs[k_parent, 0] = (  # non-employment
                    1 - prob_partner_separation_period[educ_level]  # no partner
                ) * (
                    (1 - prob_child_period) * emaxs[k_0_00, 3]  # no child
                    + prob_child_period * emaxs[k_0_10, 3]  # child
                ) + (
                    prob_partner_separation_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_0_01, 3]  # no child
                        + prob_child_period * emaxs[k_0_11, 3]  # child
                    )
                )

                emaxs[k_parent, 1] = (  # part-time employment
                    1 - prob_partner_separation_period[educ_level]  # no partner
                ) * (
                    (1 - prob_child_period) * emaxs[k_1_00, 3]  # no child
                    + prob_child_period * emaxs[k_1_10, 3]  # child
                ) + (
                    prob_partner_separation_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_1_01, 3]  # no child
                        + prob_child_period * emaxs[k_1_11, 3]  # child
                    )
                )

                emaxs[k_parent, 2] = (
                    1 - prob_partner_separation_period[educ_level]
                ) * (  # no partner
                    (1 - prob_child_period) * emaxs[k_2_00, 3]  # no child
                    + prob_child_period * emaxs[k_2_10, 3]  # child
                ) + (
                    prob_partner_separation_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_2_01, 3]  # no child
                        + prob_child_period * emaxs[k_2_11, 3]  # child
                    )
                )

            else:
                # Partner is not present in the parent (current) state, i.e.,
                # partner arrives or does not arrive in the child (future) state
                emaxs[k_parent, 0] = (  # non-employment
                    1 - prob_partner_arrival_period[educ_level]  # no partner
                ) * (
                    (1 - prob_child_period) * emaxs[k_0_00, 3]  # no child
                    + prob_child_period * emaxs[k_0_10, 3]  # child
                ) + (
                    prob_partner_arrival_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_0_01, 3]  # no child
                        + prob_child_period * emaxs[k_0_11, 3]  # child
                    )
                )

                emaxs[k_parent, 1] = (  # part-time employment
                    1 - prob_partner_arrival_period[educ_level]  # no partner
                ) * (
                    (1 - prob_child_period) * emaxs[k_1_00, 3]  # no child
                    + prob_child_period * emaxs[k_1_10, 3]  # child
                ) + (
                    prob_partner_arrival_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_1_01, 3]  # no child
                        + prob_child_period * emaxs[k_1_11, 3]  # child
                    )
                )

                emaxs[k_parent, 2] = (
                    1 - prob_partner_arrival_period[educ_level]
                ) * (  # no partner
                    (1 - prob_child_period) * emaxs[k_2_00, 3]  # no child
                    + prob_child_period * emaxs[k_2_10, 3]  # child
                ) + (
                    prob_partner_arrival_period[educ_level]  # partner
                    * (
                        (1 - prob_child_period) * emaxs[k_2_01, 3]  # no child
                        + prob_child_period * emaxs[k_2_11, 3]  # child
                    )
                )

        else:
            # Child not possible
            if partner_indicator == 1:
                # Partner is present in the parent (current) state, i.e.,
                # partner remains or is lost in the child (future) state
                emaxs[k_parent, 0] = (
                    1 - prob_partner_separation_period[educ_level]
                ) * emaxs[
                    k_0_00, 3
                ] + prob_partner_separation_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_0_01, 3
                ]  # partner
                emaxs[k_parent, 1] = (
                    1 - prob_partner_separation_period[educ_level]
                ) * emaxs[
                    k_1_00, 3
                ] + prob_partner_separation_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_1_01, 3
                ]  # partner
                emaxs[k_parent, 2] = (
                    1 - prob_partner_separation_period[educ_level]
                ) * emaxs[
                    k_2_00, 3
                ] + prob_partner_separation_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_2_01, 3
                ]  # partner

            else:
                # Partner is not present in the parent (current) state, i.e.,
                # partner arrives or does not arrive in the child (future) state
                emaxs[k_parent, 0] = (
                    1 - prob_partner_arrival_period[educ_level]
                ) * emaxs[
                    k_0_00, 3
                ] + prob_partner_arrival_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_0_01, 3
                ]  # partner
                emaxs[k_parent, 1] = (
                    1 - prob_partner_arrival_period[educ_level]
                ) * emaxs[
                    k_1_00, 3
                ] + prob_partner_arrival_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_1_01, 3
                ]  # partner
                emaxs[k_parent, 2] = (
                    1 - prob_partner_arrival_period[educ_level]
                ) * emaxs[
                    k_2_00, 3
                ] + prob_partner_arrival_period[  # no partner
                    educ_level
                ] * emaxs[
                    k_2_01, 3
                ]  # partner

    return emaxs


@numba.njit
def _get_max_aggregated_utilities(
    delta,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    emaxs,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    male_wage,
    child_benefits,
    equivalence,
):
    current_max_value_function = INVALID_FLOAT

    for j in range(NUM_CHOICES):

        if j == 0:
            consumption = non_employment_consumption_resources / equivalence
        else:
            household_income = (
                hours[j] * np.exp(log_wage_systematic + draws[j - 1]) + male_wage
            )
            deductions = calculate_deductions(deductions_spec, household_income)
            taxable_income = household_income - deductions

            tax = calculate_tax(income_tax_spec, taxable_income)

            consumption = (taxable_income - tax + child_benefits) / equivalence

        consumption_utility = consumption ** mu / mu

        value_function_choice = (
            consumption_utility * non_consumption_utilities[j] + delta * emaxs[j]
        )

        if value_function_choice > current_max_value_function:
            current_max_value_function = value_function_choice

    return current_max_value_function


@numba.guvectorize(
    ["f8, f8, f8[:], f8[:, :], f8[:], f8[:], f8, f8, f8[:], f8[:], f8, f8, f8, f8[:]"],
    "(), (), (n_choices), (n_draws, n_emp_choices), (n_choices), (n_choices), (), (), (n_spec_params), (n_spec_params), (), (), () -> ()",
    nopython=True,
    target="parallel",
)
def construct_emax(
    delta,
    log_wage_systematic,
    non_consumption_utilities,
    draws,
    emaxs,
    hours,
    mu,
    non_employment_consumption_resources,
    deductions_spec,
    income_tax_spec,
    male_wage,
    child_benefits,
    equivalence,
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

    emax[0] = 0.0

    for i in range(num_draws):
        max_total_utility = _get_max_aggregated_utilities(
            delta,
            log_wage_systematic,
            non_consumption_utilities,
            draws[i],
            emaxs,
            hours,
            mu,
            non_employment_consumption_resources,
            deductions_spec,
            income_tax_spec,
            male_wage,
            child_benefits,
            equivalence,
        )

        emax[0] += max_total_utility

    emax[0] /= num_draws
