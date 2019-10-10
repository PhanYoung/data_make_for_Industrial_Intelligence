import numpy as np


def quintile(a, ignore=False):
    if ignore:
        return np.percentile(a, [2, 25, 50, 75,98])
    return np.percentile(a, [0, 25, 50, 75, 100])

