import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def load_data(num_samples, num_class=2):
    """
    Generate a synthetic dataset with random features and class labels.

    Parameters:
    - num_samples (int): The number of data points to generate.
    - num_class (int, optional): The number of classes (default is 2).

    Returns:
    - X (numpy.ndarray): The feature matrix.
    - y (numpy.ndarray): The array of class labels.
    """
    X = np.random.randint(0, 10, size=(num_samples, 2))
    y = np.random.randint(0, num_class, size=num_samples)
    return X, y

def test_data(n_samples):
    """
    Generate a synthetic dataset using make_blobs from scikit-learn.

    Parameters:
    - n_samples (int): The number of data points to generate.

    Returns:
    - X (numpy.ndarray): The feature matrix.
    - y (numpy.ndarray): The array of class labels.
    """
    centers = [(np.random.randint(1, 10), np.random.randint(1, 10)), (np.random.randint(1, 10), np.random.randint(1, 10))]
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=np.random.randint(1, 100))
    return X, y

class KNNClassifier(object):
    """
    K-nearest neighbors (KNN) classifier.

    Parameters:
    - k (int, optional): The number of neighbors to consider (default is 3).

    Methods:
    - fit(X, y): Fit the KNN model to the training data.
    - predict(X): Predict class labels for a given set of data points.
    - plot(X, y, h=0.1): Plot the decision boundaries of the KNN classifier.
    """
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Parameters:
        - k (int, optional): The number of neighbors to consider (default is 3).
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the KNN model to the training data.

        Parameters:
        - X (numpy.ndarray): The feature matrix of the training data.
        - y (numpy.ndarray): The array of class labels.
        """
        self.X = X
        self.y = y

    def l2_distance(self, x1, x2):
        """
        Compute the L2 (Euclidean) distance between two data points.

        Parameters:
        - x1 (numpy.ndarray): The first data point.
        - x2 (numpy.ndarray): The second data point.

        Returns:
        - distance (float): The L2 distance between x1 and x2.
        """
        return np.linalg.norm(x1 - x2)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict class labels for a given set of data points.

        Parameters:
        - X (numpy.ndarray): The feature matrix of data points to predict.

        Returns:
        - y_pred (numpy.ndarray): The predicted class labels.
        """
        distance_btw_neighbors = [self.l2_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distance_btw_neighbors)[1:self.k+1]
        k_nearest_labels = [self.y[i] for i in k_indices]
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]

    def plot(self, X, y, h=0.1):
        """
        Predict the class label for a single data point.

        Parameters:
        - x (numpy.ndarray): The data point to predict.

        Returns:
        - label (int): The predicted class label.
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN Decision Boundary (k={self.k})')
        plt.show()

if __name__ == "__main__":
    num_samples, classes, k = 250, 2, 11
    X, y = test_data(num_samples)
    knn = KNNClassifier(k=k)
    knn.fit(X, y)
    knn.plot(X, y)