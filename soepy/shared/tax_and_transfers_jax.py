import jax.numpy as jnp


def calculate_net_income(
    income_tax_spec, deductions_spec, female_wage, male_wage, tax_splitting=True
):
    """Calculate income tax and soli based on income. This function does not separate
    between spousal splitting and individual income. It just applies the german tax
    function.

    Parameters
    ----------
    income_tax_spec : jnp.ndarray
        Shape (4, 4) tax specification matrix
    deductions_spec : jnp.ndarray
        Shape (2,) deduction specification [rate, cap]
    female_wage : float
        Female gross wage
    male_wage : float
        Male gross wage
    tax_splitting : bool (static)
        Whether to apply Ehegattensplitting (spousal splitting)

    Returns
    -------
    net_income : float
        Net income after taxes and deductions
    """
    female_deductions = calculate_ssc_deductions(deductions_spec, female_wage)
    male_deductions = calculate_ssc_deductions(deductions_spec, male_wage)

    # Calculate both scenarios
    taxable_income_spouse = (
        male_wage + female_wage - male_deductions - female_deductions
    )
    taxable_income_single = female_wage - female_deductions

    # Calculate income tax for spouse scenario
    if tax_splitting:
        inc_tax_spouse = (
            calculate_inc_tax(income_tax_spec, taxable_income_spouse / 2) * 2
        )
    else:
        inc_tax_spouse = calculate_inc_tax(
            income_tax_spec, female_wage - female_deductions
        ) + calculate_inc_tax(income_tax_spec, male_wage - male_deductions)

    # Calculate income tax for single scenario
    inc_tax_single = calculate_inc_tax(income_tax_spec, taxable_income_single)

    # Use boolean masking to select appropriate scenario
    has_spouse = male_wage > 0

    taxable_income = (
        has_spouse * taxable_income_spouse + (1 - has_spouse) * taxable_income_single
    )
    inc_tax = has_spouse * inc_tax_spouse + (1 - has_spouse) * inc_tax_single

    # Add soli (5% surcharge on income tax)
    net_income = taxable_income - inc_tax * 1.05

    return net_income


def calculate_inc_tax(tax_params, taxable_income):
    """Calculates the income tax.

    Parameters
    ----------
    tax_params : jnp.ndarray
        Shape (4, 4) matrix where:
        - Row 0: thresholds
        - Row 1: base tax amounts
        - Row 2: linear coefficients
        - Row 3: quadratic coefficients
    taxable_income : float
        Taxable income amount

    Returns
    -------
    tax_rate : float
        Calculated tax amount
    """
    thresholds = tax_params[0, :]
    base_amounts = tax_params[1, :]
    linear_coefs = tax_params[2, :]
    quadratic_coefs = tax_params[3, :]

    # Create boolean masks for each interval
    in_interval_0 = (taxable_income >= thresholds[0]) & (taxable_income < thresholds[1])
    in_interval_1 = (taxable_income >= thresholds[1]) & (taxable_income < thresholds[2])
    in_interval_2 = (taxable_income >= thresholds[2]) & (taxable_income < thresholds[3])
    in_interval_3 = taxable_income >= thresholds[3]

    # Calculate tax for each interval
    def calc_interval_tax(interval_idx):
        difference = taxable_income - thresholds[interval_idx]
        tax_from_formula = (
            linear_coefs[interval_idx] * difference
            + quadratic_coefs[interval_idx] * difference**2
        )
        return base_amounts[interval_idx] + tax_from_formula

    # Select the appropriate tax calculation based on which interval we're in
    tax_rate = (
        in_interval_0 * calc_interval_tax(0)
        + in_interval_1 * calc_interval_tax(1)
        + in_interval_2 * calc_interval_tax(2)
        + in_interval_3 * calc_interval_tax(3)
    )

    # Apply zero tax if below first threshold
    above_threshold = taxable_income >= thresholds[0]
    tax_rate = above_threshold * tax_rate

    return tax_rate


def calculate_ssc_deductions(deductions_spec, gross_labor_income):
    """Determines the social security contribution amount
    to be deduced from the individuals gross labor income.

    Parameters
    ----------
    deductions_spec : jnp.ndarray
        Shape (2,) array with [contribution_rate, income_cap]
    gross_labor_income : float
        Gross labor income

    Returns
    -------
    ssc_contrib : float
        Social security contribution amount
    """
    capped_income = jnp.minimum(gross_labor_income, deductions_spec[1])
    ssc_contrib = deductions_spec[0] * capped_income

    return ssc_contrib
