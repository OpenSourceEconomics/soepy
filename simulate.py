import pandas as pd
from soepy.python.simulate.simulate_python import simulate

# Generate simulated dataset
data_frame = simulate('toy_model_init_file_1000.yml') 

# Save data frame to csv file
data_frame.to_csv('toy_model_sim_test_1000.csv', sep = '\t')