"""
TODO: Functions to load datasets from scikit-learn for Linear Regression exercises
"""

from sklearn.datasets import make_regression, load_boston

def load_synthetic_regression(n_samples=100, n_features=1, noise=0.1, random_state=None):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    return X, y

def load_bostom_dataset():
    data = load_boston()
    X = data.data
    y = data.target
    return X, y