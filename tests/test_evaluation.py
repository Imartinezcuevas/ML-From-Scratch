"""
TODO: Unit tests for evaluation functions
"""

import numpy as np
from ml.evaluation import mean_squared_error, r2_score

def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    mse = mean_squared_error(y_true, y_pred)
    assert mse == 0

def test_r2():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    r2 = r2_score(y_true, y_pred)
    assert r2 == 1