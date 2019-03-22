import yaml

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
print(file_list[3])

with open(file_list[3]) as y:
    init_dict = yaml.load(y)

print(init_dict["PARAMETERS"])

# Generate simulated dataset
data_frame = simulate(file_list[3])

# Save data frame to csv file
data_frame.to_csv("test.soepy.csv", sep="\t")

cleanup()
