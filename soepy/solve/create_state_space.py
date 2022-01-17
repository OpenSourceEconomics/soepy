import numba
import numpy as np

from soepy.shared.shared_constants import MISSING_INT
from soepy.shared.shared_constants import NUM_CHOICES


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
        model_spec.num_periods + model_spec.init_exp_max,
        model_spec.num_periods + model_spec.init_exp_max,
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
                        for exp_f in range(
                            model_spec.num_periods + model_spec.init_exp_max + 1
                        ):

                            # Loop over all admissible years of experience accumulated
                            # in part-time
                            for exp_p in range(
                                model_spec.num_periods + model_spec.init_exp_max + 1
                            ):

                                # The accumulation of experience cannot exceed time elapsed
                                # since individual entered the model
                                if (
                                    exp_f + exp_p
                                    > period
                                    + model_spec.init_exp_max * 2
                                    - model_spec.educ_years[educ_level]
                                ):
                                    continue

                                if exp_f > period + model_spec.init_exp_max:
                                    continue

                                if exp_p > period + model_spec.init_exp_max:
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
                                        exp_p,
                                        exp_f,
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
                                        exp_p,
                                        exp_f,
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
                                            + model_spec.init_exp_max
                                            - model_spec.educ_years[educ_level]
                                        ):
                                            continue

                                        # If individual has only worked part-time in the past,
                                        # she can only have part-time (1) as lagged choice
                                        if (choice_lagged != 1) and (
                                            exp_p
                                            == period
                                            + model_spec.init_exp_max
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
                                            + 2 * model_spec.init_exp_max
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
