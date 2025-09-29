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

def accuracy_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp + tn) / (tp + tn + fp + fn)

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)

def f1(y_true, y_pred):
    recallV = recall(y_true, y_pred)
    precisionV = precision(y_true, y_pred)
    return (recallV * precisionV) / (precisionV + recallV)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(((1-y_true) * np.log(1 - y_pred)) + (y_true * np.log(y_pred)))
    return loss