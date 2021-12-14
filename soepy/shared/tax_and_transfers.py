import numba


@numba.jit(nopython=True)
def calculate_tax(income_tax_spec, taxable_income, male_wage, tax_splitting=True):
    """Calculate income tax and soli based on income. This function does not separate
    between spousal splitting and individual income. It just applies the german tax
    function."""

    if tax_splitting:
        # Ehegattensplitting
        if male_wage > 0:
            inc_tax = calculate_inc_tax(income_tax_spec, taxable_income / 2) * 2
        else:
            inc_tax = calculate_inc_tax(income_tax_spec, taxable_income)

    else:
        inc_tax = calculate_inc_tax(
            income_tax_spec, taxable_income
        ) + calculate_inc_tax(income_tax_spec, male_wage)

    # Add soli
    return inc_tax * (1 + income_tax_spec[4])


@numba.jit(nopython=True)
def calculate_inc_tax(income_tax_spec, taxable_income):
    """Calculates the income tax."""
    tax_base = taxable_income - income_tax_spec[0]

    if taxable_income < income_tax_spec[0]:
        tax_rate = 0
    elif (taxable_income >= income_tax_spec[0]) and (
        taxable_income < income_tax_spec[1]
    ):
        tax_rate = (
            (income_tax_spec[3] - income_tax_spec[2])
            / (income_tax_spec[1] - income_tax_spec[0])
            / 2
        ) * tax_base ** 2 + income_tax_spec[2] * tax_base
    else:
        tax_rate = (
            (income_tax_spec[3] + income_tax_spec[2])
            * (income_tax_spec[1] - income_tax_spec[0])
            / 2
        ) + income_tax_spec[3] * (taxable_income - income_tax_spec[1])

    # This does not make sense for me.
    return tax_rate
