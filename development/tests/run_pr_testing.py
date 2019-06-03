#!/usr/bin/env python
"""This script allows us to run some more extensive testing for our pull requests."""
import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# We specify our baseline intensity here.
num_minutes_robustness = 0.1
num_hours_property = 5
num_tests_regression = 100

# We want to run a very short test battery on TRAVIS to ensure the script is still
# intact.
if os.getenv("TRAVIS") is not None:
    print(" \n ... using TRAVIS specification")
    num_minutes_robustness = 0.1
    num_hours_property = 0.01
    num_tests_regression = 1

print(" \n ... running robustness tests")
cmd = "python run.py {:}".format(num_minutes_robustness)
subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/robustness")

print(" \n ... running regression tests")
cmd = ""
cmd += "python run.py --request check --num {:}".format(num_tests_regression)
subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/regression")

print(" \n ... running property tests")
cmd = "python run.py --request run --hours {:}".format(num_hours_property)
subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/property")
