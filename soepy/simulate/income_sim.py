import numba

from soepy.shared.non_employment import calc_erziehungsgeld
from soepy.shared.tax_and_transfers import calculate_net_income


@numba.guvectorize(
    ["f8[:], f8[:, :], f8[:], f8, b1, f8[:]"],
    "(n_ssc_params), (n_tax_params, n_tax_params), (num_work_choices), (), () -> (num_work_choices)",
    nopython=True,
    target="cpu",
    # target="parallel",
)
def calculate_employment_consumption_resources(
    deductions_spec,
    income_tax_spec,
    current_female_income,
    male_wage,
    tax_splitting,
    employment_consumption_resources,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to be employed.
    It adds the components from the budget constraint to the female wage."""

    for choice_num in range(current_female_income.shape[0]):
        employment_consumption_resources[choice_num] = calculate_net_income(
            income_tax_spec,
            deductions_spec,
            current_female_income[choice_num],
            male_wage,
            tax_splitting,
        )


@numba.guvectorize(
    ["f8, f8, b1, b1, f8, f8, f8, f8[:]"],
    "(), (), (), (), (), (), () -> ()",
    nopython=True,
    target="cpu",
)
def calculate_erziehungsgeld_vector(
    male_wage,
    female_income,
    married,
    baby_child,
    erziehungsgeld_inc_single,
    erziehungsgeld_inc_married,
    erziehungsgeld,
    erziehungsgeld_calculated,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to be employed.
    It adds the components from the budget constraint to the female wage."""

    erziehungsgeld_calculated[0] = calc_erziehungsgeld(
        male_wage,
        female_income,
        married,
        baby_child,
        erziehungsgeld_inc_single,
        erziehungsgeld_inc_married,
        erziehungsgeld,
    )
