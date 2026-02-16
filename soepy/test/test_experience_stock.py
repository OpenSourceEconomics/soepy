import numpy as np

from soepy.shared.experience_stock import exp_years_to_stock
from soepy.shared.experience_stock import max_exp_years
from soepy.shared.experience_stock import next_stock
from soepy.shared.experience_stock import stock_to_exp_years


def test_max_exp_years_uses_larger_of_period_and_pt_scaled_period():
    init_exp_max = 4

    assert max_exp_years(period=10, init_exp_max=init_exp_max) == 18


def test_stock_years_roundtrip():
    init_exp_max = 4
    period = 10

    stock = 0.25
    exp_years = stock_to_exp_years(
        stock=stock,
        period=period,
        init_exp_max=init_exp_max,
    )
    stock_back = exp_years_to_stock(
        exp_years=exp_years,
        period=period,
        init_exp_max=init_exp_max,
    )

    np.testing.assert_allclose(stock_back, stock)


def test_next_stock_full_time_and_part_time_increments():
    init_exp_max = 4
    period = 0

    stock0 = 0.0

    model_params = type(
        "Params",
        (),
        {"gamma_p": np.array([0.5]), "gamma_p_mom": 0.0},
    )()

    # At period 1: max = 2 * 4 + 1 = 9
    stock_ft = next_stock(
        stock=stock0,
        period=period,
        init_exp_max=init_exp_max,
        choice=2,
        model_params=model_params,
        educ_level=np.array([0]),
        child_age=np.array([-1]),
        biased_exp=False,
    )
    np.testing.assert_allclose(stock_ft, 1.0 / 9.0)

    stock_pt = next_stock(
        stock=stock0,
        period=period,
        init_exp_max=init_exp_max,
        choice=1,
        model_params=model_params,
        educ_level=np.array([0]),
        child_age=np.array([-1]),
        biased_exp=False,
    )
    np.testing.assert_allclose(stock_pt, 0.5 / 9.0)


def test_next_stock_clips_to_unit_interval():
    init_exp_max = 0
    period = 5

    model_params = type(
        "Params",
        (),
        {"gamma_p": np.array([0.5]), "gamma_p_mom": 0.0},
    )()

    stock = 1.0
    stock_next = next_stock(
        stock=stock,
        period=period,
        init_exp_max=init_exp_max,
        choice=2,
        model_params=model_params,
        educ_level=np.array([0]),
        child_age=np.array([-1]),
        biased_exp=False,
    )

    assert 0.0 <= stock_next <= 1.0
