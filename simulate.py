import pandas as pd
from soepy.python.simulate.pyth_simulate import simulate

# Generate simulated dataset
data_frame = simulate('toy_model_init_file_large.yml') 

# Save data frame to csv file
data_frame.to_csv('toy_model_sim_test_large.csv', sep = '\t')