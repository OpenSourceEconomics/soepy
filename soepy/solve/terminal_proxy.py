import jax
import jax.numpy as jnp

from soepy.shared.constants_and_indices import AGE_YOUNGEST_CHILD
from soepy.shared.constants_and_indices import EDUC_LEVEL
from soepy.shared.constants_and_indices import NUM_CHOICES
from soepy.shared.experience_stock import next_stock
from soepy.shared.wages import calculate_log_wage


def terminal_proxy_continuation(
    exp_grid,
    states_period,
    covariates_period,
    model_params,
    model_spec,
    biased_exp,
    current_period,
):
    """Compute terminal proxy continuation values.

    Parameters
    ----------
    exp_grid : jax.numpy.ndarray, shape (n_grid,)
        Experience grid for continuous experience.
    states_period : jax.numpy.ndarray, shape (n_states, n_state_vars)
        State array for the current period.
    covariates_period : jax.numpy.ndarray, shape (n_states, n_covariates)
        Covariates array for the current period.

    Returns
    -------
    jax.numpy.ndarray
        Proxy continuation values with shape (n_states, NUM_CHOICES, n_grid).
    """
    edu_state = states_period[:, EDUC_LEVEL]
    child_age = states_period[:, AGE_YOUNGEST_CHILD]
    male_wage = covariates_period[:, 1]
    log_male = jnp.log(male_wage)

    choice_ids = jnp.arange(NUM_CHOICES)

    def per_state(educ_level_state, child_age_state):
        def per_choice(choice):
            x_next = next_stock(
                stock=exp_grid,
                period=current_period,
                init_exp_max=model_spec.init_exp_max,
                choice=choice,
                model_params=model_params,
                educ_level=educ_level_state,
                child_age=child_age_state,
                biased_exp=biased_exp,
            )

            log_w_female = calculate_log_wage(
                model_params=model_params,
                educ=educ_level_state,
                period=current_period + 1,
                init_exp_max=model_spec.init_exp_max,
                exp_stock=x_next,
            ) + jnp.log(model_spec.elasticity_scale)

            return log_w_female

        return jax.vmap(per_choice)(choice_ids)

    log_w_female = jax.vmap(per_state)(edu_state, child_age)

    proxy = -jnp.exp(
        model_params.beta_0
        + model_params.beta_1 * log_w_female
        + model_params.beta_3 * log_male[:, None, None]
    )

    return proxy
