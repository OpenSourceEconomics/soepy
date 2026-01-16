import jax
import numpy as np

from soepy.shared.tax_and_transfers_jax import calculate_net_income


def calculate_employment_consumption_resources(
    model_spec,
    current_female_income,
    male_wage,
    tax_splitting,
):
    """This function calculates the resources available to the individual
    to spend on consumption were she to choose to be employed.
    It adds the components from the budget constraint to the female wage."""

    employment_consumption_resources = jax.vmap(
        calculate_net_income, in_axes=(None, None, 1, None, None)
    )(
        model_spec.tax_params,
        model_spec.ssc_deductions,
        current_female_income,
        male_wage,
        tax_splitting,
    )

    return np.array(np.transpose(employment_consumption_resources), copy=True)
