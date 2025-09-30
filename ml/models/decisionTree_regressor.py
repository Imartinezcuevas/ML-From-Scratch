import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        """
        Store tree build recursively using _build_tree
        """
        self.n_samples, self.n_features = X.shape
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        TODO:
        - Check stopping conditions (max_depth, min_samples_split, or pure node)
        - If stopping, create leaf node with mean of y
        - Otherwise:
            - Find best feature and threshold (_find_best_split)
            - Split X, y into left and right subsets
            - Recursively build left and right nodes
            - Return Node with split info
        """
        if depth >= self.max_depth or len(y) < self.min_samples_split or X.shape[0] < self.min_samples_split:
            leaf_value = np.mean(y)
            return self.Node(value=leaf_value)
        
        if len(np.unique(y)) == 1:
            return self.Node(value=y[0])

        best_feature, best_threshold = self._find_best_split(X, y)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold 

        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return self.Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left = left_node,
            right = right_node
        )


    def _find_best_split(self, X, y):
        """
        TODO:
        - Find the best feature and threshold to split X and y
        - Steps:
            1. Initialize best_score to infinity (minimize variance)
            2. Loop over each feature
            3. For each feature, loop over canditate thresholds. Threshold can be unique values or midpoints
            4. For each threshold:
                - Split y into left and right subsets
                - Compute weighted variance of left and right
                - Compute score = weighted variance
                - If score < best_score, update best_feature, best_threshold, best_score
                5. Return best_feature and best_threshold
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')

        for feature in range(self.n_features):
            sorted_vals = np.sort(np.unique(X[:, feature]))
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                y_left = y[left_idx]
                y_right = y[right_idx]
                left_var =  np.var(y_left)
                right_var = np.var(y_right)
                n_left, n_right = len(y_left), len(y_right)
                n_total = n_left + n_right
                score = (n_left / n_total) * left_var + (n_right / n_total) * right_var
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold

    def predict(self, X):
        """
        TODO:
        - Transverse the tree for each sample using _traverse_tree
        """
        predictions = []
        for x in X:
            predictions.append(self._traverse_tree(x, self.tree))
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        """
        TODO:
        - If leaf node: return value
        - Otherwise, follow left or right child depending on the threshold
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)