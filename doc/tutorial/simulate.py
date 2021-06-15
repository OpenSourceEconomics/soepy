import soepy
from development.tests.auxiliary.auxiliary import cleanup

# Generate simulated dataset
data_frame = soepy.simulate(
    "model_params.pkl", "model_spec_init.yml", is_expected=False
)

cleanup()
