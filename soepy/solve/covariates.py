import numpy as np
import pandas as pd

from soepy.shared.shared_constants import HOURS


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
    married = states[:, 7] == 1
    no_child = states[:, 6] == -1

    # Age youngest child
    # Bins of age of youngest child based on kids age
    # bin 0 corresponds to no kid, remaining bins as in Blundell
    # 0-2, 3-5, 6-10, 11+
    age_kid = pd.Series(states[:, 6])
    bins = pd.cut(
        age_kid, bins=[-2, -1, 2, 5, 10, 11], labels=[0, 1, 2, 3, 4],
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
    male_wages = np.where(married, np.exp(log_wages) * HOURS[2], 0)

    equivalence_scale = create_equivalence_scale(no_child, married)

    # Child benefits
    # If a woman has a child she receives child benefits
    child_benefits = np.where(states[:, 6] == -1, 0, model_spec.child_benefits)

    # Collect in covariates vector
    covariates = np.column_stack((bins, male_wages, equivalence_scale, child_benefits))

    return covariates


def create_equivalence_scale(no_child, married):
    """Depending on the presence of a partner and a child each state is assigned an
    equivalence scale value following the modernized OECD scale: 1 for a single woman
    HH, 1.5 for a woman with a partner, 1.8 for a woman with a partner and a child and
    1.3 for a woman with a child and no partner"""
    equivalence_scale = np.where(no_child & ~married, 1.0, np.nan)
    equivalence_scale = np.where(no_child & married, 1.5, equivalence_scale)
    equivalence_scale = np.where(~no_child & married, 1.8, equivalence_scale)
    equivalence_scale = np.where(~no_child & ~married, 1.3, equivalence_scale)

    assert (
        np.isnan(equivalence_scale).any() == 0
    ), "Some HH were not assigned an equivalence scale"
    return equivalence_scale
