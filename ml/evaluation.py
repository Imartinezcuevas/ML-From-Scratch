"""
TODO: Functions to evaluate Linear Regression predictions
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    TODO:
    - Compute mean squared error between true and predicted values
    """
    # Assert si no tienen las mismas dimensiones
    return np.square(np.subtract(y_true, y_pred)).mean()
    

def r2_score(y_true, y_pred):
    """
    TODO:
    - Compute R^2 score
    """
    pass