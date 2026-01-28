import numpy as np

from soepy.shared.interpolation import linear_interp_1d


def test_linear_interp_exact_on_grid_points():
    grid = np.linspace(0.0, 1.0, 5)
    values = grid**2

    for x in grid:
        out = linear_interp_1d(grid=grid, values=values, x=x)
        np.testing.assert_allclose(out, x**2)


def test_linear_interp_midpoint():
    grid = np.array([0.0, 1.0])
    values = np.array([0.0, 2.0])

    out = linear_interp_1d(grid=grid, values=values, x=0.25)
    np.testing.assert_allclose(out, 0.5)


def test_linear_interp_vectorized_x():
    grid = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 2.0, 4.0])
    x = np.array([0.5, 1.5])

    out = linear_interp_1d(grid=grid, values=values, x=x)
    np.testing.assert_allclose(out, np.array([1.0, 3.0]))
