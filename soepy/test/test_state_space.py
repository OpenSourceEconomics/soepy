import collections

import numpy as np
import pytest

from soepy.solve.create_state_space import create_state_space_objects


@pytest.fixture()
def model_spec():
    spec = {
        "num_periods": 3,
        "num_educ_levels": 3,
        "num_types": 2,
        "child_age_max": 10,
        "partner_cf_const": 0.0,
        "partner_cf_age": 0.0,
        "partner_cf_age_sq": 0.0,
        "partner_cf_educ": 0.0,
        "child_benefits": 0.0,
    }
    return collections.namedtuple("model_specification", spec.keys())(**spec)


def test_state_space_shapes(model_spec):
    (
        states,
        indexer,
        covariates,
        child_age_update_rule,
        child_state_indexes,
    ) = create_state_space_objects(model_spec=model_spec)

    assert states.shape[1] == 6
    assert covariates.shape[0] == states.shape[0]
    assert child_age_update_rule.shape[0] == states.shape[0]
    assert child_state_indexes.shape[0] == states.shape[0]

    # Partner indicator implies positive male wage.
    np.testing.assert_array_equal(states[:, 5] == 1, covariates[:, 1] > 0)

    # Child bin non-zero implies child present.
    np.testing.assert_array_equal(covariates[:, 0] != 0, states[:, 4] > -1)
