import soepy
from development.tests.auxiliary.auxiliary import cleanup

# Generate simulated dataset
data_frame = soepy.simulate(
    "toy_model_init_file_04_3types.pkl", "model_spec_init_01.yml", is_expected=False
)

cleanup()
