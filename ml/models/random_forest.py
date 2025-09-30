import numpy as np
from decisionTree_classifier import DecisionTreeClassifier
from decisionTree_regressor import DecisionTreeRegressor

class RandomForest:
    def __init__(self, n_trees: int, max_depth: int = None, min_samples_split: int = 2, max_features=None, task:str = "classification", random_state:int = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.random_state = random_state
    
        np.random.seed(self.random_state)
        self.trees = []
        self.n_features = None
        self.classes_ = None # to map labels if needed

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        if self.task == "classification":
            self.classes_= np.unique(y)
            self.label_map = {label: i for i, label in enumerate(self.classes_)}
            y_mapped = np.array([self.label_map[label] for label in y])
        else:
            y_mapped = y

        if self.max_features is None:
            if self.task == "Classification":
                self.max_features = int(np.sqrt(self.n_features))
            else:
                self.max_features = max(1, self.n_features // 3)

        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Create a new tree
            if self.task == "classification":
                tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.max_features)
            else:
                tree = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for i in range(X.shape[0]):
            sample_preds = all_preds[:, i]
            if self.task == "classification":
                result = np.bincount(sample_preds).argmax()
            else:
                result = np.mean(sample_preds)
            final_preds.append(result)

        return np.array(final_preds)
    

