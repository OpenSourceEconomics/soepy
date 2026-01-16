def calculate_net_income(
    income_tax_spec, deductions_spec, female_wage, male_wage, tax_splitting=True
):
    """Calculate income tax and soli based on income. This function does not separate
    between spousal splitting and individual income. It just applies the german tax
    function."""

    female_deductions = calculate_ssc_deductions(deductions_spec, female_wage)

    # Check if spouse is present
    if male_wage > 0:
        male_deductions = calculate_ssc_deductions(deductions_spec, male_wage)
        taxable_income = male_wage + female_wage - male_deductions - female_deductions
        if tax_splitting:
            # Ehegattensplitting

            inc_tax = calculate_inc_tax(income_tax_spec, taxable_income / 2) * 2
        else:
            inc_tax = calculate_inc_tax(
                income_tax_spec, female_wage - female_deductions
            ) + calculate_inc_tax(income_tax_spec, male_wage - male_deductions)

    else:
        taxable_income = female_wage - female_deductions
        inc_tax = calculate_inc_tax(income_tax_spec, taxable_income)

    # Add soli
    net_income = taxable_income - inc_tax * 1.05

    return net_income


def calculate_inc_tax(tax_params, taxable_income):
    """Calculates the income tax."""
    thresholds = tax_params[0, :]
    if taxable_income < thresholds[0]:
        tax_rate = 0
    else:
        if (taxable_income >= thresholds[0]) and (taxable_income < thresholds[1]):
            interval_num = 0
        elif (taxable_income >= thresholds[1]) and (taxable_income < thresholds[2]):
            interval_num = 1
        elif (taxable_income >= thresholds[2]) and (taxable_income < thresholds[3]):
            interval_num = 2
        else:
            interval_num = 3

        difference_to_calc = taxable_income - tax_params[0, interval_num]
        tax_rate = (
            tax_params[2, interval_num] * difference_to_calc
            + tax_params[3, interval_num] * difference_to_calc**2
        )
        tax_rate += tax_params[1, interval_num]

    return tax_rate


def calculate_ssc_deductions(deductions_spec, gross_labor_income):
    """Determines the social security contribution amount
    to be deduced from the individuals gross labor income"""

    capped_income = min(gross_labor_income, deductions_spec[1])

    ssc_contrib = deductions_spec[0] * capped_income

    return ssc_contrib
