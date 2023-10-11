import matplotlib.pyplot as plt
import numpy as np

def load_data(num_samples, n_features, seed=0):
    np.random.seed(seed)
    X = np.random.rand(num_samples, n_features)
    y = np.random.randint(2, size=(num_samples))
    return X, y 

def test_data(num_samples, n_features, seed=0):
    np.random.seed(0)

    # Generate synthetic data
    num_samples = num_samples
    num_features = n_features
    X = np.random.rand(num_samples, num_features)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        if np.sqrt(X[i, 0]**2 + X[i, 1]**2) > 0.6:
            y[i] = 1
    return X, y

def plot_decision_boundary(X, y, decision_tree, title="Decision Tree Boundary"):
    # Set the plot range
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Generate a grid of points for the plot
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Make predictions for each point in the grid
    Z = np.array([decision_tree.predict([xi, yi]) for xi, yi in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Create the contour plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r', edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

class Tree(object):
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, depth=0, label=None) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.label = label

class DecisionTree(object):

    def __init__(self, max_depth):
        self.tree = Tree()
        self.max_depth = max_depth

    def _predict(self, x, tree):
        if tree.threshold is None:
            return tree.label
        if x[tree.feature_idx] <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def predict(self, x):
        return self._predict(x, self.tree)

    def entropy(self, y):
        p1 = np.sum(y) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        return - (p0 * np.log2(p0)) - (p1 * np.log2(p1))

    def information_gain(self, y, y_left, y_right):
        H_y = self.entropy(y)
        H_y_left = self.entropy(y_left)
        H_y_right = self.entropy(y_right)
        p_left = sum(y_left) / len(y_left)
        p_right = sum(y_right) / len(y_right)
        return H_y - (p_left * H_y_left + p_right * H_y_right) #Information gain = H(s) - sum(|s_i| / |s| * H(s_i)) where i is index for a partition set s_i

    def split(self, X, y):
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
        d_tree.feature_idx = feature_idx
        d_tree.threshold = threshold
        d_tree.left = Tree(depth=depth + 1)
        d_tree.right = Tree(depth=depth + 1)

    def fit(self, X, y, d_tree, depth=0):
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
    decision_tree.fit(X, y, decision_tree.tree)
    x_pred = np.array([0.2, 0.2])
    y_pred= decision_tree.predict(x_pred)
    plot_decision_boundary(X, y, decision_tree)
    