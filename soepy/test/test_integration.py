from soepy.python.simulate.simulate_python import simulate
from soepy.test.random_init import random_init


def test_1():
    """This test makes sure the full package works for random initialization files."""
    random_init()

    simulate("test.soepy.yml")
