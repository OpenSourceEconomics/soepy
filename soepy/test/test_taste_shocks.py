import numpy as np

from soepy.solve.emaxs import _expected_max_with_taste_shocks


def test_expected_max_with_taste_shocks_matches_stable_formula():
    choice_values = np.array([1.0, 3.0, -2.0])
    lambda_taste = 0.25

    max_value = np.max(choice_values)
    expected = max_value + lambda_taste * np.log(
        np.sum(np.exp((choice_values - max_value) / lambda_taste))
    )

    calculated = _expected_max_with_taste_shocks(choice_values, lambda_taste)

    np.testing.assert_allclose(calculated, expected)


def test_expected_max_with_taste_shocks_is_numerically_stable():
    choice_values = np.array([1000.0, 1001.0, 999.0])
    lambda_taste = 0.5

    calculated = _expected_max_with_taste_shocks(choice_values, lambda_taste)

    assert np.isfinite(calculated)
    assert calculated > np.max(choice_values)
