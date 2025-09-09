"""
TODO: Functions to evaluate Linear Regression predictions
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()
    

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)