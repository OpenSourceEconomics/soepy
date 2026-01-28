import collections

import numpy as np
import pytest

from soepy.solve.create_state_space import create_state_space_objects


@pytest.fixture()
def model_spec():
    spec = {
        "num_periods": 2,
        "num_educ_levels": 3,
        "num_types": 1,
        "child_age_max": 10,
        "partner_cf_const": 0.0,
        "partner_cf_age": 0.0,
        "partner_cf_age_sq": 0.0,
        "partner_cf_educ": 0.0,
        "child_benefits": 0.0,
    }
    return collections.namedtuple("model_specification", spec.keys())(**spec)


def test_state_space_basic_smoke(model_spec):
    states, indexer, covariates, *_ = create_state_space_objects(model_spec=model_spec)

    assert states.shape[1] == 6
    assert covariates.shape[0] == states.shape[0]
    assert indexer.ndim == 6

    assert np.isin(states[:, 2], [0, 1, 2]).all()
