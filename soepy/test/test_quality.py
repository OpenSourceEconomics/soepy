"""This module contains test that check the code quality of the package."""
import subprocess

import numpy as np

from soepy.python.soepy_config import PACKAGE_DIR


def test1():
    """This test runs flake8 to ensure the code quality."""
    np.testing.assert_equal(subprocess.call(["flake8"], cwd=PACKAGE_DIR), 0)
