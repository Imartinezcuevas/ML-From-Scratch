# TODO
# 1. What a linear model is and how it represents relationships between variables.
# 2. What a loss function is and why it is necessary.
# 3. How model parameters are optimized (concept or gradient descent).
# 4. How to evaluate a regression model (which metrics are commonly used).
import numpy as np
from ml.evaluation import mean_squared_error

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None 
        self.loss_history = []
        self.params_history = []
        self.test_loss_history = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        TODO:
        - Initialize weights and bias
        - Loop over n_iters:
            - Compute predictions
            - Compute gradients
            - Update weights and bias
        """
        self.n_samples, self.n_features = np.shape(X)
        # 1. Initialize weights self.W as zeros of shape (n_features,)
        self.W = np.zeros(shape=(self.n_features,))
        # 2. Initialize bias self.b as 0
        self.b = 0

        # 3. Loop over number of iterations
        for i in range(self.n_iters):
            # a. Compute predictions using curring weights and bias
            y_pred = X.dot(self.W) + self.b

            # b. Calculate gradients
            dW = (1 / self.n_samples) * X.T.dot(y_pred - y)
            db = (1 / self.n_samples) * np.sum(y_pred - y)

            # c. Update parameters
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

            # d. Compute and save the current loss
            loss = mean_squared_error(y, y_pred)
            self.loss_history.append(loss)
            self.params_history.append((self.W.copy(), self.b))

            if X_val is not None and y_val is not None:
                y_pred_test = X_val.dot(self.W) + self.b
                loss_test = mean_squared_error(y_val, y_pred_test)
                self.test_loss_history.append(loss_test)

    def predict(self, X):
        """
        TODO:
        - Compute linear prediction: XW + b
        """
        return X.dot(self.W) + self.b
