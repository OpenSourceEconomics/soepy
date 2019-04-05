"""This module provides some configuration for the package."""
import sys
from pathlib import Path

import numpy as np

import soepy

# We only support modern Python.
np.testing.assert_equal(sys.version_info[:2] >= (3, 6), True)

# We rely on relative paths throughout the package.
ROOT_DIR = Path(soepy.__path__[0])
PACKAGE_DIR = ROOT_DIR.parents[0]
TEST_RESOURCES_DIR = PACKAGE_DIR / "soepy" / "test" / "resources"
