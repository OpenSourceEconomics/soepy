from functools import partial

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(5,))
def calculate_net_income(
    income_tax_spec,
    deductions_spec,
    female_wage,
    male_wage,
    partner_indicator,
    tax_splitting,
):
    """Calculate income tax and soli based on income. This function does not separate
    between spousal splitting and individual income. It just applies the german tax
    function."""
    female_deductions = calculate_ssc_deductions(deductions_spec, female_wage)

    # Check if spouse is present
    male_deductions = calculate_ssc_deductions(deductions_spec, male_wage)
    taxable_income = male_wage + female_wage - male_deductions - female_deductions
    if tax_splitting:
        # Ehegattensplitting
        inc_tax = calculate_inc_tax(
            income_tax_spec, taxable_income / (partner_indicator + 1)
        ) * (partner_indicator + 1)
    else:
        inc_tax = calculate_inc_tax(
            income_tax_spec, female_wage - female_deductions
        ) + calculate_inc_tax(income_tax_spec, male_wage - male_deductions)

    # Add soli
    net_income = taxable_income - inc_tax * 1.05

    return net_income


@jit
def calculate_inc_tax(tax_params, taxable_income):
    """Calculates the income tax."""
    thresholds = tax_params[0, :]

    interval_diff = jnp.clip(thresholds - taxable_income, a_min=0)
    interval_num = jnp.max(
        jnp.flatnonzero(
            interval_diff == jnp.min(interval_diff),
            size=thresholds.shape[0],
            fill_value=-1,
        )
    )

    difference_to_calc = taxable_income - tax_params[0, interval_num]
    tax_rate = (
        tax_params[2, interval_num] * difference_to_calc
        + tax_params[3, interval_num] * difference_to_calc ** 2
    ) + tax_params[1, interval_num]
    return tax_rate


@jit
def calculate_ssc_deductions(deductions_spec, gross_labor_income):
    """Determines the social security contribution amount
    to be deduced from the individuals gross labor income"""

    capped_income = jnp.minimum(gross_labor_income, deductions_spec[1])

    ssc_contrib = deductions_spec[0] * capped_income

    return ssc_contrib
