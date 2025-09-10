import numpy as np
from ml.evaluation import binary_cross_entropy

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
        self.loss_history = []

    def sigmoid(self, z):
        """
        TODO:
        - Implement sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        TODO:
        - Initialize weights and bias
        - Loop over n_iters:
            - Compute linear predictions
            - Apply sigmoid -> probabilities
            - Compute loss
            - Compute gradients
            - Update weights and bias
        """
        self.n_samples, self.n_features = np.shape(X)
        # 1. Initialize weights self.W as zeros of shape (n_features,)
        self.W = np.zeros(shape=(self.n_features,))
        # 2. Initialize bias as zero
        self.b = 0

        # 3. Loop over number of iterations
        for _ in range(self.n_iters):
            # a. Compute predictions
            y_pred_proba = self.sigmoid(X.dot(self.W) + self.b)

            # b. Compute loss and save it
            loss = binary_cross_entropy(y, y_pred_proba)
            self.loss_history.append(loss)

            # c. Calculate gradients
            dW = (1/self.n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1/self.n_samples) * np.sum(y_pred_proba - y)

            # d. Update params
            self.W = self.W - self.lr*dW
            self.b = self.b - self.lr*db


    def predict_proba(self, X):
        """
        TODO:
        - Return predicted probabilities using sigmoid.
        """
        return self.sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        """
        TODO:
        - Convert predicted probabilities to classes (0 and 1)
        - Use threshold 0.5
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= 0.5).astype(int)
