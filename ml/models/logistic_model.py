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
        - Implementar la función sigmoide: 1 / (1 + exp(-z))
        - Esto convierte la salida lineal en una probabilidad (0 a 1).
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        TODO:
        - Inicializar pesos y bias
        - Loop sobre n_iters:
            - Calcular la combinación lineal (XW + b)
            - Aplicar sigmoide → probabilidades
            - Calcular pérdida (binary cross-entropy)
            - Calcular gradientes
            - Actualizar pesos y bias
        """
        self.n_samples, self.n_features = np.shape(X)
        self.W = np.zeros(shape=(self.n_features,))
        self.b = 0

        for _ in range(self.n_iters):
            y_pred_proba = self.sigmoid(X.dot(self.W) + self.b)

            loss = binary_cross_entropy(y, y_pred_proba)
            self.loss_history.append(loss)

            dW = (1/self.n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1/self.n_samples) * np.sum(y_pred_proba - y)

            self.W = self.W - self.lr*dW
            self.b = self.b - self.lr*db


    def predict_proba(self, X):
        """
        TODO:
        - Retornar las probabilidades predichas usando la sigmoide.
        """
        return self.sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        """
        TODO:
        - Convertir las probabilidades en clases (0 o 1)
        - Usa un umbral de 0.5
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= 0.5).astype(int)
