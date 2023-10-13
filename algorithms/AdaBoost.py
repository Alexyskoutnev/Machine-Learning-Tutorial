from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_synthetic_dataset(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=None):
    """
    Generate a synthetic dataset for AdaBoost.

    Parameters:
    - n_samples: Number of data points in the dataset.
    - n_features: Number of features.
    - n_informative: Number of informative features (the features that contribute to class separation).
    - n_redundant: Number of redundant features (features that are derived from informative features).
    - random_state: Random seed for reproducibility.

    Returns:
    - X: The feature matrix.
    - y: The target labels.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state
    )
    return X, y

class Adaboost(object):

    def __init__(self, n_estimators=50):
        """
        Initialize an AdaBoost classifier.

        Parameters:
        - n_estimators: Number of AdaBoost iterations (weak classifiers).
        """
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        """
        Fit the AdaBoost ensemble on the training data.

        Parameters:
        - X: Training data feature matrix.
        - y: Training data target labels.
        """
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            predictions = stump.predict(X)
            error = sum(w[y != predictions]) / sum(w)
            alpha = 0.5 * np.log((1  - error) / (error + 1e-10))
            w = w * np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.alphas.append(alpha)
            self.stumps.append(stump)

    def predict(self, X):
        """
        Make predictions on new data using the trained AdaBoost ensemble.

        Parameters:
        - X: New data feature matrix.

        Returns:
        - Predicted class labels.
        """
        n_samples = X.shape[0]
        stump_prediction = np.array([stump.predict(X) for stump in self.stumps])
        weighted_prediction = np.dot(self.alphas, stump_prediction)
        return np.sign(weighted_prediction)

    def plot(self, X, y):
        """
        Plot the decision boundary of the AdaBoost ensemble.

        Parameters:
        - X: Feature matrix for plotting.
        - y: True class labels for plotting.
        """
        h = .1  # Step size in the mesh
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])  # Light colors for background
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])  # Bold colors for points
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Adaboost Decision Boundary")
        plt.xlabel('Feature 1')
        plt.ylabel("Feature 2")
        plt.show()

if __name__ == "__main__":
    random_seed = np.random.randint(1, 1000)
    num_samples, features = 1000, 2
    X, y = generate_synthetic_dataset(n_samples=num_samples, n_features=features, n_informative=2, n_redundant=0, random_state=random_seed)
    X_train, y_train = X[:int(num_samples*0.8)], y[:int(num_samples*0.8)]
    X_test, y_test = X[int(num_samples*0.8):], y[int(num_samples*0.8):]
    adaboost = Adaboost()
    adaboost.fit(X, y)
    y_pred = adaboost.predict(X_test)
    adaboost.plot(X_train, y_train)
