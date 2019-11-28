import soepy
from development.tests.auxiliary.auxiliary import cleanup

# Generate simulated dataset
data_frame = soepy.simulate(
    "toy_model_init_file_03_3types.pkl", "model_spec_init.yml", is_expected=False
)
data_frame = soepy.simulate(
    "toy_model_init_file_03_1type.pkl", "model_spec_init.yml", is_expected=False
)

cleanup()
