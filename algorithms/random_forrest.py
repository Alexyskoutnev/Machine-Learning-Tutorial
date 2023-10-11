import numpy as np

from decision_trees import DecisionTree, test_data, plot_decision_boundary

class RandomForrest(object):
    """
    Initialize a Random Forest classifier.

    Parameters:
    n_trees (int, optional): Number of decision trees in the forest (default is 5).
    max_depth (int, optional): Maximum depth of each decision tree (default is None, which means no maximum depth).

    Attributes:
    n_trees (int): Number of decision trees in the forest.
    max_depth (int, optional): Maximum depth set for each decision tree.
    models (list): List to store the individual decision trees in the forest.

    Methods:
    fit(X, y): Fit the Random Forest to the input data and labels.
    predict(x): Predict the class label for a given input 'x' using the Random Forest.

    Returns:
    RandomForrest: A Random Forest classifier with the specified number of trees and maximum depth.
    """

    def __init__(self, n_trees=5, max_depth=None):
        """
        Initialize a Random Forest instance.

        Parameters:
        n_trees (int, optional): Number of decision trees in the forest (default is 5).
        max_depth (int, optional): Maximum depth of each decision tree (default is None, which means no maximum depth).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        """
        Fit the Random Forest to the input data and labels.

        Parameters:
        X (numpy.ndarray): Input feature matrix with dimensions (num_samples, num_features).
        y (numpy.ndarray): Array of class labels with shape (num_samples).
        """
        for _ in range(self.n_trees):
            decision_tree = DecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(X.shape[0], X.shape[0])
            X_bootstrapped = X[indices]
            y_bootstrapped = y[indices]
            decision_tree.fit(X_bootstrapped, y_bootstrapped, decision_tree.node)
            self.models.append(decision_tree)
    
    def predict(self, x):
        """
        Predict the class label for a given input 'x' using the Random Forest.

        Parameters:
        x (array-like): The input features to classify.

        Returns:
        int: The predicted class label.
        """
        predictions = [tree.predict(x) for tree in self.models]
        return int(np.round(np.mean(predictions, axis=0)))

if __name__ == "__main__":
    num_samples, num_features = 100, 2
    max_depth = 5
    X, y = test_data(num_samples, num_features)
    decision_tree = RandomForrest(max_depth=max_depth)
    decision_tree.fit(X, y)
    plot_decision_boundary(X, y, decision_tree)