from soepy.python.simulate.simulate_python import simulate
from soepy.test.auxiliary import cleanup

# Generate simulated dataset
data_frame = simulate("toy_model_init_file_1000.yml")

# Save data frame to csv file
data_frame.to_pickle("test.soepy.pkl")

cleanup()
