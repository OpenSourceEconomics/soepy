"""The module allows to run tests from inside the interpreter."""
import os

import pytest

from soepy.python.simulate.simulate_python import simulate
from soepy.python.soepy_config import PACKAGE_DIR
import soepy.python.soepy_config


def test():
    """The function allows to run the tests from inside the interpreter."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)
    pytest.main()
    os.chdir(current_directory)
