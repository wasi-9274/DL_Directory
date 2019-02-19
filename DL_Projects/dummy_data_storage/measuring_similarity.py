import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

play_data = pd.DataFrame({'x1': [-3, -2, -1, 0, 1, 2, 3],
                          'x2': [9, 4, 1, 0, 1, 4, 9],
                          'x3': [1, 2, 3, 4, 5, 6, 7],
                          'x4': [2, 5, 15, 27, 28, 30, 31]
                          })

play_data = play_data[['x1', 'x2', 'x3', 'x4']]
print(play_data)


def pearson_corr(x, y):
    mean_x, mean_y = np.sum(x) / len(x), np.sum(x) / len(y)
    x_diffs = x - mean_x
    y_diffs = y - mean_y
    numerator = np.sum(x_diffs * y_diffs)
    denominator = np.sqrt(np.sum(x_diffs ** 2)) * np.sqrt(np.sum(y_diffs ** 2))
    corr = numerator / denominator
    return corr


def corr_spearman(x, y):
    x = x.rank()
    y = y.rank()
    mean_x, mean_y = np.sum(x) / len(x), np.sum(y) / len(y)
    x_diffs = x - mean_x
    y_diffs = y - mean_y
    numerator = np.sum(x_diffs * y_diffs)
    denominator = np.sqrt(np.sum(x_diffs ** 2)) * np.sqrt(np.sum(y_diffs ** 2))
    corr = numerator / denominator
    return corr


def kendalls_tau(x, y):
    x = x.rank()
    y = y.rank()
    n = len(x)
    sum_vals = 0
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        for j, (x_j, y_j) in enumerate(zip(x, y)):
            if i < j:
                sum_vals += np.sign(x_i - x_j)*np.sign(y_i - y_j)
    tau = 2*sum_vals/(n*(n-1))
    return tau


def eucl_dist(x, y):
    return np.linalg.norm(x - y)


def manhat_dist(x, y):
    return sum(abs(e - s) for s, e in zip(x, y))


