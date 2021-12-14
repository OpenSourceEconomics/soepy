""" This module contains the tax function from gettsim. We want to test our function
againts it."""
import numpy as np
import pandas as pd

from soepy.shared.tax_and_transfers import calculate_ssc_deductions


def piecewise_polynomial(
    x, thresholds, rates, intercepts_at_lower_thresholds, rates_multiplier=None
):
    """Calculate value of the piecewise function at `x`.

    Parameters
    ----------
    x : pd.Series
        Series with values which piecewise polynomial is applied to.
    thresholds : np.array
                A one-dimensional array containing the thresholds for all intervals.
    rates : numpy.ndarray
            A two-dimensional array where columns are interval sections and rows
            correspond to the nth polynomial.
    intercepts_at_lower_thresholds : numpy.ndarray
        The intercepts at the lower threshold of each interval.
    rates_multiplier : pd.Series, float
                       Multiplier to create individual or scaled rates. If given and
                       not equal to 1, the function also calculates new intercepts.

    Returns
    -------
    out : float
        The value of `x` under the piecewise function.

    """
    # If no individual is transferred, we return an empty series
    if x.empty:
        return x

    num_intervals = len(thresholds) - 1
    degree_polynomial = rates.shape[0]

    # Check in which interval each individual is. The thresholds are not exclusive on
    # the right side!
    binned = pd.cut(
        x,
        bins=thresholds,
        right=False,
        include_lowest=True,
        labels=range(num_intervals),
    ).astype(float)

    # Create series with last threshold for each individual
    thresholds_individual = binned.replace(dict(enumerate(thresholds[:-1])))

    # Increment for each individual in the corresponding interval
    increment_to_calc = x - thresholds_individual

    # Check if any value is in the lowest interval.
    if 0 in binned.array and intercepts_at_lower_thresholds[0] == np.nan:
        raise ValueError(f"In {x.name} is a value outside the determined range.")

    # If each individual has its own rates or the rates are scaled, we can't use the
    # intercept, which was generated in the parameter loading.
    if rates_multiplier is not None:

        # Initialize Series containing 0 for all individuals
        out = x * 0
        out += intercepts_at_lower_thresholds[0]

        # Go through all intervals except the first and last
        for i in range(2, num_intervals):
            threshold_incr = thresholds[i] - thresholds[i - 1]
            for pol in range(1, degree_polynomial + 1):

                # We only calculate the intercepts for individuals who are in this or
                # higher interval. Hence we have to use the individual rates.
                out.loc[binned >= i] += (
                    rates_multiplier.loc[binned >= i]
                    * rates[pol - 1, i - 1]
                    * threshold_incr ** pol
                )

    # If rates remain the same, everything is a lot easier.
    else:
        # We assign each individual the pre-calculated intercept.
        out = binned.replace(dict(enumerate(intercepts_at_lower_thresholds)))

    # Intialize a multiplyer for 1 if it is not given.
    rates_multiplier = 1 if rates_multiplier is None else rates_multiplier
    # Now add the evaluation of the increment
    for pol in range(1, degree_polynomial + 1):
        out += (
            binned.replace(dict(enumerate(rates[pol - 1, :])))
            * rates_multiplier
            * (increment_to_calc ** pol)
        )

    # For those in interval zero, the above equations yield wrong results
    out.loc[binned == 0] = intercepts_at_lower_thresholds[0]

    return out


def calc_gettsim_sol_individual(
    thresholds, rates, intercept_low, deductions_spec, income, soli_st
):
    deductions_ssc = calculate_ssc_deductions(deductions_spec, income)
    taxable_inc = income - deductions_ssc
    gettsim_tax = (
        piecewise_polynomial(pd.Series([taxable_inc]), thresholds, rates, intercept_low)
        * soli_st
    )
    return gettsim_tax, taxable_inc
