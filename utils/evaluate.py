import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import gridspec
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

mape = mean_absolute_percentage_error


def evaluate(y_true, y_pred, plot=True, title=None):
    err = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cor = pearsonr(y_pred, y_true)[0]
    pre = 1 - np.mean(((y_pred - y_true) / y_true) ** 2.)
    print("mse:", err, "\nmape", mape,  "\nr2:", r2, "\ncor:", cor, "\npre:", pre)
    if plot:
        plt.subplots(figsize=(8, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        plt.subplot(gs[0])
        if title:
            plt.title(title)
        plt.plot(y_true, y_pred - y_true, '.')
        plt.plot(y_true, [0]*len(y_true))
        plt.subplot(gs[1])
        if title:
            plt.title(title)
        plt.plot(y_true, y_pred, '.')
        plt.plot(y_true, y_true)

