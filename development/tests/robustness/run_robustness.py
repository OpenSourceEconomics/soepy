"""This script checks whether the package performs properly for random requests."""
import numpy as np

from soepy.python.simulate.simulate_python import simulate
from soepy.test.random_init import random_init


np.random.seed(1)
for _ in range(10):

    random_init()
    simulate("test.soepy.yml")
