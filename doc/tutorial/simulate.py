from development.tests.auxiliary.auxiliary import cleanup
from soepy.simulate.simulate_python import get_simulate_func

# Generate simulated dataset
simulate_func = get_simulate_func(
    "model_params.pkl", "model_spec_init.yml", is_expected=False, data_sparse=True
)
data_frame = simulate_func("model_params.pkl", "model_spec_init.yml")

cleanup()
