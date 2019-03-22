from pathlib import Path

from soepy.python.simulate.simulate_python import simulate
from soepy.test.auxiliary import cleanup
from soepy.python.soepy_config import TEST_RESOURCES_DIR

pathlist = Path(TEST_RESOURCES_DIR).glob("**/*.yml")
files = [x for x in pathlist if x.is_file()]

file_list = []
for file in files:
    file_list.append(str(file))

file_list.sort()

# Generate simulated dataset
data_frame = simulate(files[0])

# Save data frame to csv file
data_frame.to_csv("test.soepy.csv", sep="\t")

cleanup()
