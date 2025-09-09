"""
TODO: Unit tests for LinearRegression class
"""

import numpy as np
from ml.models.linear_model import LinearRegression

def test_fit_simple():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([3, 5, 7, 9])
    model = LinearRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)
    assert np.isclose(model.W[0], 2, 0.1)
    assert np.isclose(model.b, 3, 0.1)

def test_predict_shape():
    """
    TODO:
    - Check that predictions have the correct shape
    """
    pass