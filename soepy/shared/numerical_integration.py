import numpy as np
from scipy.special import roots_hermite

from soepy.shared.shared_auxiliary import draw_disturbances


def get_integration_draws_and_weights(model_spec, model_params):
    if model_spec.integration_method == "quadrature":
        # Draw standard points and corresponding weights
        standard_draws, draw_weights_emax = roots_hermite(model_spec.num_draws_emax)
        # Rescale draws and weights
        draws_emax = standard_draws * np.sqrt(2) * model_params.shock_sd
        draw_weights_emax *= 1 / np.sqrt(np.pi)
    elif model_spec.integration_method == "monte_carlo":
        draws_emax = draw_disturbances(
            model_spec.seed_emax, 1, model_spec.num_draws_emax, model_params
        )[0]
        draw_weights_emax = (
            np.ones(model_spec.num_draws_emax) / model_spec.num_draws_emax
        )
    else:
        raise ValueError(
            f"Integration method {model_spec.integration_method} not specified."
        )

    return draws_emax, draw_weights_emax
