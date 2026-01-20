import numpy as np
import pandas as pd

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import HOURS
from soepy.shared.constants_and_indices import PARTNER
from soepy.shared.constants_and_indices import PERIOD


def construct_covariates(states, model_spec):
    """Construct covariates depending only on the discrete state."""

    married = states[:, PARTNER] == 1
    no_child = states[:, AGE_YOUNGEST_CHILD] == -1

    age_kid = pd.Series(states[:, AGE_YOUNGEST_CHILD])
    # Last bin is "11+" in the original model. Ensure the upper edge is strictly
    # larger than 10 to avoid duplicate edges when child_age_max == 10.
    last_edge = max(int(model_spec.child_age_max), 11)

    bins = pd.cut(
        age_kid,
        bins=[-2, -1, 2, 5, 10, last_edge],
        labels=[0, 1, 2, 3, 4],
    ).to_numpy()

    log_wages = (
        model_spec.partner_cf_const
        + model_spec.partner_cf_age * states[:, PERIOD]
        + model_spec.partner_cf_age_sq * states[:, PERIOD] ** 2
        + model_spec.partner_cf_educ * states[:, EDUC_LEVEL]
    )

    male_wages = np.where(married, np.exp(log_wages) * HOURS[2], 0)

    equivalence_scale = create_equivalence_scale(no_child, married)

    child_benefits = np.where(no_child, 0, model_spec.child_benefits)

    covariates = np.column_stack((bins, male_wages, equivalence_scale, child_benefits))
    return covariates


def create_equivalence_scale(no_child, married):
    equivalence_scale = np.where(no_child & ~married, 1.0, np.nan)
    equivalence_scale = np.where(no_child & married, 1.5, equivalence_scale)
    equivalence_scale = np.where(~no_child & married, 1.8, equivalence_scale)
    equivalence_scale = np.where(~no_child & ~married, 1.3, equivalence_scale)

    assert not np.isnan(equivalence_scale).any()
    return equivalence_scale
