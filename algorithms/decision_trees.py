import matplotlib.pyplot as plt
import numpy as np

def load_data(num_samples, n_features, seed=0):
    """
    Generate random data for classification.

    Parameters:
    num_samples (int): Number of data samples to generate.
    n_features (int): Number of features for each data sample.
    seed (int, optional): Seed for random number generation.

    Returns:
    X (numpy.ndarray): Randomly generated data with dimensions (num_samples, n_features).
    y (numpy.ndarray): Randomly generated binary labels (0 or 1) with shape (num_samples).
    """
    np.random.seed(seed)
    X = np.random.rand(num_samples, n_features)
    y = np.random.randint(2, size=(num_samples))
    return X, y 

def test_data(num_samples, n_features, seed=0):
    """
    Generate synthetic data based on a simple decision boundary.

    Parameters:
    num_samples (int): Number of data samples to generate.
    n_features (int): Number of features for each data sample.
    seed (int, optional): Seed for random number generation.

    Returns:
    X (numpy.ndarray): Synthetic data with dimensions (num_samples, n_features).
    y (numpy.ndarray): Binary labels (0 or 1) based on a decision boundary with shape (num_samples).
    """
    np.random.seed(0)
    X = np.random.rand(num_samples, n_features)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        if np.sqrt(X[i, 0]**2 + X[i, 1]**2) > 0.5:
            y[i] = 1
    return X, y
    
def plot_decision_boundary(X, y, decision_tree, title="Decision Node Boundary"):
    """
    Visualize the decision boundary of the decision tree.

    Parameters:
    X (numpy.ndarray): Input data with dimensions (num_samples, 2).
    y (numpy.ndarray): Labels corresponding to input data.
    decision_tree (DecisionTree): The decision tree to visualize.
    title (str, optional): Title for the plot (default is "Decision Node Boundary").
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.array([decision_tree.predict([xi, yi]) for xi, yi in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Create the contour plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r', edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

class Node(object):
    """
    Initialize a node in the decision tree.

    Parameters:
    feature_idx (int): Index of the feature used for splitting.
    threshold (float): Threshold value for the feature.
    left (Node): Left child node.
    right (Node): Right child node.
    depth (int): Depth of the node in the tree.
    label (int, optional): Label assigned to the node if it is a leaf node.
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, depth=0, label=None) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.label = label

class DecisionTree(object):
    """
    Initialize a Decision Tree classifier.

    Parameters:
    max_depth (int): Maximum depth of the decision tree.

    Attributes:
    node (Node): Root node of the decision tree.
    max_depth (int): Maximum depth set for the decision tree.

    Methods:
    _predict(x, tree): Recursively predict the label for a given input 'x' based on the decision tree 'tree'.
    predict(x): Predict the label for a given input 'x' using the decision tree.
    entropy(y): Calculate the entropy of the target variable 'y'.
    information_gain(y, y_left, y_right): Calculate the information gain by splitting the dataset based on 'y', 'y_left', and 'y_right'.
    split(X, y): Find the best feature and threshold to split the dataset 'X' and 'y'.
    _update(d_tree, feature_idx, threshold, depth): Update a decision tree node 'd_tree' with feature 'feature_idx' and 'threshold'.
    fit(X, y, d_tree, depth): Recursively build a decision tree for the dataset 'X' and 'y' and attach it to the node 'd_tree' with a specified depth.

    Returns:
    DecisionTree: A decision tree classifier with the specified maximum depth.
    """
    def __init__(self, max_depth):
        """
        Initialize a decision tree instance.

        Parameters:
        max_depth (int): The maximum depth of the decision tree.
        """
        self.node = Node()
        self.max_depth = max_depth

    def _predict(self, x, tree):
        """
        Predict the class label for a given input.

        Parameters:
        x (array-like): The input features to classify.
        tree (Node): The root node of the decision tree.

        Returns:
        int: The predicted class label.
        """
        if tree.threshold is None:
            return tree.label
        if x[tree.feature_idx] <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def predict(self, x):
        """
        Predict the class label for a given input.

        Parameters:
        x (array-like): The input features to classify.

        Returns:
        int: The predicted class label.
        """
        return self._predict(x, self.node)

    def entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        y (array-like): The array of class labels.

        Returns:
        float: The entropy of the labels.
        """
        p1 = np.sum(y) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        return - (p0 * np.log2(p0)) - (p1 * np.log2(p1))

    def information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain when splitting a dataset.

        Parameters:
        y (array-like): The array of original class labels.
        y_left (array-like): The array of class labels in the left subset.
        y_right (array-like): The array of class labels in the right subset.

        Returns:
        float: The information gain of the split.
        """
        H_y = self.entropy(y)
        H_y_left = self.entropy(y_left)
        H_y_right = self.entropy(y_right)
        p_left = sum(y_left) / len(y_left)
        p_right = sum(y_right) / len(y_right)
        return H_y - (p_left * H_y_left + p_right * H_y_right) #Information gain = H(s) - sum(|s_i| / |s| * H(s_i)) where i is index for a partition set s_i

    def split(self, X, y):
        """
        Find the best feature and threshold to split the dataset.

        Parameters:
        X (array-like): The input feature matrix.
        y (array-like): The array of class labels.

        Returns:
        int: The index of the best feature for splitting.
        float: The threshold value for the best split.
        array: Boolean mask for the left subset.
        """
        max_gain = 0.0
        best_threshold = None
        n, m = X.shape
        for feature_idx in range(m):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            for i in range(1, len(unique_values)):
                threshold = (unique_values[i - 1] + unique_values[i]) / 2.0 #take the average between a feature value to make decision boundary
                y_left = y[feature_values <= threshold]
                y_right = y[feature_values > threshold]
                gain = self.information_gain(y, y_left, y_right)
                if gain > max_gain:
                    max_gain = gain
                    best_threshold = threshold
                    best_feature_idx = feature_idx
                    left_mask = feature_values <= threshold
        return best_feature_idx, best_threshold, left_mask

    def _update(self, d_tree, feature_idx, threshold, depth):
        """
        Update the current node with splitting information.

        Parameters:
        d_tree (Node): The current node to update.
        feature_idx (int): Index of the best feature used for splitting.
        threshold (float): Threshold value for the feature.
        depth (int): Depth of the current node.
        """
        d_tree.feature_idx = feature_idx
        d_tree.threshold = threshold
        d_tree.left = Node(depth=depth + 1)
        d_tree.right = Node(depth=depth + 1)

    def fit(self, X, y, d_tree, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        X (array-like): The input feature matrix.
        y (array-like): The array of class labels.
        d_tree (Node): The current node to split.
        depth (int): Depth of the current node.

        Returns:
        Node: The root node of the decision tree.
        """
        if depth == self.max_depth or np.all(y == y[0]): #stop case for recursion if all the data contains the same labels
            d_tree.label = int(np.mean(y))
            return
        best_feature_idx, best_threshold, left_mask = self.split(X, y)
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        self._update(d_tree, best_feature_idx, best_threshold, depth)
        self.fit(X_left, y_left, d_tree.left, depth + 1)
        self.fit(X_right, y_right, d_tree.right, depth + 1)
        return d_tree

if __name__ == "__main__":
    num_samples, num_features = 1000, 2
    max_depth = 5
    X, y = test_data(num_samples, num_features)
    decision_tree = DecisionTree(max_depth=max_depth)
    decision_tree.fit(X, y, decision_tree.node)
    plot_decision_boundary(X, y, decision_tree)
    