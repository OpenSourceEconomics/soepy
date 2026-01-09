import numpy as np
from scipy.special import roots_hermite


def get_integration_draws_and_weights(model_spec):
    if model_spec.integration_method == "quadrature":
        # Draw standard points and corresponding weights
        standard_draws, draw_weights_emax = roots_hermite(model_spec.num_draws_emax)
        # Rescale draws and weights
        draws_emax = standard_draws * np.sqrt(2)
        draw_weights_emax *= 1 / np.sqrt(np.pi)
    elif model_spec.integration_method == "monte_carlo":
        draws_emax = draw_zero_one_distributed_shocks(
            model_spec.seed_emax, 1, model_spec.num_draws_emax
        )[0]
        draw_weights_emax = (
            np.ones(model_spec.num_draws_emax) / model_spec.num_draws_emax
        )
    else:
        raise ValueError(
            f"Integration method {model_spec.integration_method} not specified."
        )

    return draws_emax, draw_weights_emax


def draw_zero_one_distributed_shocks(seed, num_periods, num_draws):
    """Creates desired number of draws of a multivariate standard normal
    distribution.

    """
    np.random.seed(seed)

    mean = 0

    # Create draws from the standard normal distribution
    draws = np.random.normal(mean, 1, (num_periods, num_draws))

    return draws
