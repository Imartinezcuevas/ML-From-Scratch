"""
TODO: Functions to load datasets from scikit-learn for Linear Regression exercises
"""

from sklearn.datasets import make_regression

def load_synthetic_regression(n_samples=100, n_features=1, noise=0.1, random_state=None, bias=None):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, bias=bias)
    return X, y, bias