#!/usr/bin/env python
"""This script checks whether the package performs properly for random requests."""
import datetime
import sys

from soepy.simulate.simulate_python import simulate
from soepy.test.random_init import random_init


def func(maxrt):
    stop = datetime.datetime.now() + maxrt
    while datetime.datetime.now() < stop:
        random_init()
        simulate("test.soepy.pkl", "test.soepy.yml")


if __name__ == "__main__":

    minutes = float(sys.argv[1])
    func(datetime.timedelta(minutes=0.1))
