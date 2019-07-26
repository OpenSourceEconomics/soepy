import soepy
from development.tests.auxiliary.auxiliary import cleanup

# Generate simulated dataset
data_frame = soepy.simulate("toy_model_init_file_02_1000_3types.yml")

cleanup()
