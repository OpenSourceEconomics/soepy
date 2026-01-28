# Continuous Experience Refactor (Summary)

This document summarizes the changes applied so far to migrate `soepy` from a
(discrete) full-time/part-time experience state representation to a single
continuous experience stock.

## Model / State Space

- Discrete experience dimensions removed.
- New discrete state vector layout (6 columns):
  - `period`
  - `educ_level`
  - `lagged_choice`
  - `type`
  - `age_youngest_child`
  - `partner`
- Column indices are centralized in `soepy/shared/state_space_indices.py`.

Files:
- `soepy/shared/state_space_indices.py`
- `soepy/solve/create_state_space.py`

## Continuous Experience Stock

- Experience is represented as a stock `x ∈ [0, 1]`.
- Period-specific scaling uses
  - `max_exp_years(period) = init_exp_max + max(period, period * pt_increment)`
- Experience accumulation:
  - non-employment: `+0`
  - part-time: `+pt_increment`
  - full-time: `+1`
- Expected vs actual law of motion:
  - `is_expected=True`: `pt_increment = gamma_p_bias`
  - else: `pt_increment = pt_exp_ratio`

Files:
- `soepy/shared/experience_stock.py`

## Interpolation

- Added a minimal 1D linear interpolation helper for values on a 1D grid.
- Guideline: avoid using `jnp.asarray` inside JAX code; assume JAX arrays.

Files:
- `soepy/shared/interpolation.py`
- `AGENTS.md`

## Continuation Values: Interpolate Then Aggregate

- Implemented required ordering for the EMAX recursion:
  1. interpolate continuation values on the experience grid
  2. aggregate over child/partner probabilities
- Implementation is intentionally low-dimensional and readable via `vmap`.

Files:
- `soepy/solve/continuous_continuation.py`

## EMAX / Solve Output Shape

- Solver now produces `emaxs` with shape:
  - `(n_states, n_grid, n_choices + 1)`
- `n_grid` defaults to `10` via `model_spec.experience_grid_points`.

Files:
- `soepy/solve/solve_python.py`
- `soepy/solve/emaxs.py`

## Wages

- Wage equation is now continuous-experience only:
  - single return to experience (reusing `gamma_f` as the slope)
  - wage depends on `log(exp_years + 1)`
  - expectation bias is handled in experience accumulation (not in wages)

Files:
- `soepy/shared/wages.py`

## Non-employment / Resources

- `non_employment` functions were updated to broadcast correctly when the wage input
  is on the experience grid (`(n_states, n_grid)`).

Files:
- `soepy/shared/non_employment.py`

## Simulation (Refactor: Continuous Stock)

- Simulation state now carries `Experience_Stock` instead of PT/FT experience.
- `emaxs` and wage/resources are interpolated from the grid to each agent’s stock.
- Initial experience years are drawn from legacy PT/FT share files by convolution,
  then mapped to the stock.
- Initial `lagged_choice` rule:
  - `2` if initial experience years `> 1`, else `0`.

Files:
- `soepy/simulate/constants_sim.py`
- `soepy/simulate/simulate_auxiliary.py`
- `soepy/simulate/simulate_python.py`
- `soepy/exogenous_processes/experience.py`
- `soepy/exogenous_processes/determine_lagged_choice.py`

## Tests

Added / updated (continuous-only):
- `soepy/test/test_experience_stock.py`
- `soepy/test/test_interpolation.py`
- `soepy/test/test_continuous_continuation.py`
- `soepy/test/test_full_solve_continuous.py` (5-period full solve vs explicit reference DP)
- `soepy/test/test_child_index.py` (child transition indexer consistency)

Adjusted/skipped because they depended on discrete experience regression targets:
- `soepy/test/test_regression.py`
- `soepy/test/test_single_woman.py`

## Notes / Follow-ups

- CI-level checks are intentionally left to the user (`pytest`, `pre-commit`).
- Legacy regression-vault expectations are not comparable after this refactor; they
  need to be regenerated under the continuous model if you want regression testing.
