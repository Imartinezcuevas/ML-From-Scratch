import numpy as np
from ml.evaluation import accuracy_score

class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", p=1.2):
        """
        TODO:
        - k: number of neighbors to consider
        """
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        TODO:
        - Save training data
        - In kNN there isn't parameters to optimize, we just save data
        """
        self.X_train = X
        self.y_train = y

    def _compute_distance(self, x1, x2):
        """
        TODO:
        - Implement a function to compute the distance between two points
        """
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.metric == "manhattan":
            return  np.sum(abs(x1 - x2))
        elif self.metric == "minkowski":
            return  np.sum(abs(x1 - x2)**self.p)**(1/self.p)
        elif self.metric == "chebyshev":
            return np.max(abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")

    def _get_neighbors(self, x):
        """
        TODO:
        - Calculate the distance from x to all points in the training set
        - Order by distance
        - Return the closest k neighbors 
        """
        distances = []
        for i, neighbor in enumerate(self.X_train):
            distance = self._compute_distance(x, neighbor)
            distances.append([distance, i])

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        indices = [index for (_, index) in k_nearest]
        return self.y_train[indices]

    def predict(self, X):
        """
        TODO:
        - For each point of X:
            - Get the k nearest neighbors
            - Perfom a mojority vote to determine the predicted class
        - Return an array with the predicted classes
        """
        labels = []
        for x in X:
            # Get the labels from the k_neighbors
            k_nearest = self._get_neighbors(x)
            # Count frequencies and find the label with the max count
            counter = {}
            maxfLabel = -1
            maxf = 0
            for k in k_nearest:
                counter[k] = 1 + counter.get(k, 0)
                if counter[k] > maxf:
                    maxfLabel = k
                    maxf = counter[k]
            labels.append(maxfLabel)
        return np.array(labels)
