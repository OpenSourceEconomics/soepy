"""This module contains test that check the code quality of the package."""
import subprocess
import os

from soepy.python.soepy_config import PACKAGE_DIR
from subprocess import CalledProcessError


def test1():
    """This test runs flake8 to ensure the code quality. However, this is only relevant during
    development."""
    try:
        import flake8  # noqa: F401
    except ImportError:
        return None

    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR)
    try:
        subprocess.check_call(["flake8"])
        os.chdir(cwd)
    except CalledProcessError:
        os.chdir(cwd)
        raise CalledProcessError
