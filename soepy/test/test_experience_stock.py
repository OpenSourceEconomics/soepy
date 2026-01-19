import numpy as np

from soepy.shared.experience_stock import exp_years_to_stock
from soepy.shared.experience_stock import max_exp_years
from soepy.shared.experience_stock import next_stock
from soepy.shared.experience_stock import stock_to_exp_years


def test_max_exp_years_uses_larger_of_period_and_pt_scaled_period():
    init_exp_max = 4

    assert max_exp_years(period=10, init_exp_max=init_exp_max, pt_increment=0.5) == 14
    assert max_exp_years(period=10, init_exp_max=init_exp_max, pt_increment=2.0) == 24


def test_stock_years_roundtrip():
    init_exp_max = 4
    period = 10
    pt_increment = 0.5

    stock = 0.25
    exp_years = stock_to_exp_years(
        stock=stock,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )
    stock_back = exp_years_to_stock(
        exp_years=exp_years,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
    )

    np.testing.assert_allclose(stock_back, stock)


def test_next_stock_full_time_and_part_time_increments():
    init_exp_max = 4
    period = 0
    pt_increment = 0.5

    stock0 = 0.0

    # At period 1: max = 4 + max(1, 0.5) = 5
    stock_ft = next_stock(
        stock=stock0,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
        choice=2,
    )
    np.testing.assert_allclose(stock_ft, 1.0 / 5.0)

    stock_pt = next_stock(
        stock=stock0,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
        choice=1,
    )
    np.testing.assert_allclose(stock_pt, 0.5 / 5.0)


def test_next_stock_clips_to_unit_interval():
    init_exp_max = 0
    period = 5
    pt_increment = 0.5

    stock = 1.0
    stock_next = next_stock(
        stock=stock,
        period=period,
        init_exp_max=init_exp_max,
        pt_increment=pt_increment,
        choice=2,
    )

    assert 0.0 <= stock_next <= 1.0
