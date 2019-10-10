import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr


import random
random.seed(715)


class SampleSet(object):
    '''
        generate a sample dataset to get random normal distibuted data with min-max limited
    '''
    def __init__(self, meanv, stdv, size, minv=None, maxv=None):
        candset = SampleSet.gen_candset(meanv, stdv, size)
        self.candset = SampleSet.trim_candset(candset, minv, maxv)

    @staticmethod
    def gen_candset(meanv, stdv, num):
        return np.random.randn(num) * stdv + meanv

    @staticmethod
    def trim_candset(candset, minv, maxv):
        r = candset
        if maxv is not None:
            r = r[r <= maxv]
        if minv is not None:
            r = r[r >= minv]
        return r

    def get_samples(self, num=None):
        return np.random.choice(self.candset, num)




def normalized_trans(d, trans_func, **trans_paras):
    scaler = MinMaxScaler()
    dn = scaler.fit_transform(d.reshape(-1, 1))
    dt = trans_func(dn, **trans_paras)
    rf = np.cos(-(6.2*dt) ** 3 / 15.)
    dt = dt + 0.07 * rf * (1-dt)
    dr = scaler.inverse_transform(dt).reshape(-1)
    return dr



def normalized_trans2(d1, d2, trans_func):
    scaler1, scaler2 = MinMaxScaler(), MinMaxScaler()
    dn1 = scaler1.fit_transform(d1.reshape(-1, 1)).reshape(-1)
    dn2 = scaler2.fit_transform(d2.reshape(-1, 1)).reshape(-1)
    dt = trans_func(dn1, dn2)
    dr = scaler1.inverse_transform(dt.reshape(-1,1)).reshape(-1)
    return dr



def normalized_transx(d1, trans_func, *dx, **trans_paras):
    scaler1 = MinMaxScaler()
    dn1 = scaler1.fit_transform(d1.reshape(-1, 1))
    sc_list = []
    dn_list = []
    for d in dx:
        ss = MinMaxScaler()
        dd = ss.fit_transform(d.reshape(-1, 1))
        sc_list.append(ss)
        dn_list.append(dd)

    dt = trans_func(dn1, dn_list, **trans_paras)
    dr = scaler1.inverse_transform(dt).reshape(-1)
    return dr


def with_normalization(func):
    def trans_func(d1, dl, **kwargs):
        func(d1, dl, **kwargs)
        pass
    return trans_func





def nonlinearize(a, kl):
    return sum([a**(i+1) * v for i, v in enumerate(kl)])


from functools import reduce
def trans(d0, dl, ul, kl):
    ''' 
    dl: list of dataset,
    ul: list of bias
    kl: list of ref
    '''
    c = [u - (abs(k - d)) **3 for d, u, k in zip(dl, ul, kl)]
    r = reduce(lambda x, y: x*y, c)
    return r * d0



def amplify_diff(d1, d2, rate=0.5):
    diff = d1 - d2
    signs = np.ones(len(d1))
    signs[diff < 0] = -1
    new_diff = abs(diff) ** 0.5 * signs * rate
    return d1 + new_diff



def add_nois(a, error_range=None, rate=0.1):
    if not error_range:
        error_range = a.std() * rate
    return a + error_range * np.random.randn(len(a))


def random_acc(mass_in, acc_start=0, diff_rate=0.01):
    acc = acc_start
    for i in mass_in:
        acc = acc + i
        out = (diff_rate * np.random.randn() + 1) * acc
        yield out
        acc -= out


def minmax_scale(a):
    return MinMaxScaler().fit_transform(a.reshape(-1,1)).reshape(-1)


def conditional_acc(mass_in, condition, acc_start=0, decay_rate=0.05, need_scale=True):
    if need_scale:
        condition = minmax_scale(condition) - 0.5
    acc = acc_start
    for i, c in zip(mass_in, condition):
        acc += i
        out = (decay_rate * c + 1) * acc
        yield out
        acc -= out

