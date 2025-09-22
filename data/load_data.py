"""
TODO: Functions to load datasets from scikit-learn for Linear Regression exercises
"""

from sklearn.datasets import make_regression, make_classification

def load_synthetic_regression(n_samples=100, n_features=1, noise=0.1, random_state=None, bias=None):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, bias=bias)
    return X, y, bias

def load_synthetic_classification(n_samples=100, n_features=1, n_informative=None, n_redundant=0, n_repeated=0, n_classes=2, class_sep=1.0, flip_y=0.0, random_state=None):
    if n_informative is None:
        n_informative = n_features

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, class_sep=class_sep, flip_y=flip_y, random_state=random_state)

    return X, y