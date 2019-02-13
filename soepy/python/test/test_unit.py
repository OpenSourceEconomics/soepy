
import numpy as np

from soepy.python.simulate.simulate_python import simulate
from soepy.python.shared.shared_auxiliary import calculate_wage_systematic
from soepy.python.shared.shared_auxiliary import calculate_period_wages
from soepy.python.pre_processing.model_processing import read_init_file
from soepy.python.test.random_init import random_init


def test1():
    """"""
    for _ in range(10):

        init_dict = random_init()

        optim_paras = init_dict['PARAMETERS']['optim_paras']

        educ_level = np.array([0.,0.,0.])
        educ_level_index = np.random.choice([0,1,2])
        educ_level[educ_level_index] = 1.

        exp_p, exp_f = np.random.randint(0, 10, 2)

        stat = calculate_wage_systematic(educ_level, exp_p, exp_f, optim_paras)

        # Construct wage components
        gamma_s0 = np.dot(educ_level, optim_paras[0:3])
        gamma_s1 = np.dot(educ_level, optim_paras[3:6])
        period_exp_sum = exp_p * np.dot(educ_level, optim_paras[6:9]) + exp_f
        depreciation = 1 - np.dot(educ_level, optim_paras[9:12])

        # Calculate wage in the given state
        period_exp_total = period_exp_sum * depreciation + 1
        returns_to_exp = gamma_s1 * period_exp_total
        wage_systematic = np.exp(gamma_s0) * returns_to_exp

        np.testing.assert_array_almost_equal(stat, wage_systematic)

def test2():
    """"""
    init_dict = random_init()
    attr_dict = read_init_file('test.soepy.yml')
    optim_paras = init_dict['PARAMETERS']['optim_paras']

    educ_level = np.array([0., 0., 0.])
    educ_level_index = np.random.choice([0, 1, 2])
    educ_level[educ_level_index] = 1.

    wage_systematic = calculate_wage_systematic(educ_level, exp_p, exp_f, optim_paras)

    period_wages = calculate_period_wages(attr_dict, wage_systematic, 100)

    np.testing.assert_array_almost_equal(period_wages, wage_systematic * np.exp(100))


def test3():
    """"""
    random_init()
    df_test = simulate('test.soepy.yml')


    for year in [11, 12, 13, 14]:
        df = df[(df['Years of Education'] == year) & (df[df['Period'] < year -10])]
        df = df[[col for col in df.columns.values if col not in ['Identifier', 'Periode']]]

        assert df_test.isnull().all()

