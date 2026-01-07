"""This module provides the fixtures for the PYTEST runs."""
import os
import tempfile

import jax
import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1223)
    # THis is always called. Just update the config here. As good as anywhere else.
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def fresh_directory():
    """Each test is executed in a fresh directory."""
    os.chdir(tempfile.mkdtemp())
