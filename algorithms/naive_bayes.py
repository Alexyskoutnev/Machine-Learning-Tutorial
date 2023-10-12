import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

class GaussianBayesClassifer(object):
    """
    A custom Gaussian Naive Bayes classifier with fit, predict, and plot methods.

    Attributes:
    - class_priors (dict): A dictionary storing class priors.
    - class_means (dict): A dictionary storing class means.
    - class_variances (dict): A dictionary storing class variances.

    Methods:
    - __init__(self): Initializes the classifier with empty class priors, means, and variances.
    - _prior(self, X_c, X): Calculates the prior probability for a class.
    - fit(self, X, y): Fits the classifier to the provided training data.
    - predict(self, X): Predicts class labels for a given set of feature vectors.
    - _predict(self, x): Predicts the class label for a single feature vector.
    - _pdf(self, _class, x): Computes the probability density function for a feature vector in a given class.
    - plot(self, X, y, h=0.1): Generates a decision boundary plot for the classifier.

    """
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

    def _prior(self, X_c, X):
        """
        Calculate the prior probability for a class.

        Parameters:
        - X_c (numpy.ndarray): Feature vectors for a specific class.
        - X (numpy.ndarray): The entire feature matrix.

        Returns:
        - float: Prior probability for the class.
        """
        return len(X_c) / len(X)

    def fit(self, X, y):
        """
        Fit the classifier to the provided training data.

        Parameters:
        - X (numpy.ndarray): The feature matrix.
        - y (numpy.ndarray): The array of class labels.
        """
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[c == y]
            self.class_priors[c] = self._prior(X_c, X)
            self.class_means[c] = X_c.mean(axis=0)
            self.class_variances[c] = X_c.var(axis=0)

    def predict(self, X):
        """
        Predict class labels for a set of feature vectors.

        Parameters:
        - X (numpy.ndarray): Feature vectors to predict.

        Returns:
        - numpy.ndarray: Predicted class labels.
        """
        _pred = np.array([self._predict(x) for x in X])
        return _pred

    def _predict(self, x):
        """
        Predict the class label for a single feature vector.

        Parameters:
        - x (numpy.ndarray): A single feature vector.

        Returns:
        - int: Predicted class label.
        """
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.class_priors[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posteriors[c] = prior + likelihood
        return max(posteriors, key=posteriors.get)

    def _pdf(self, _class, x):
        """
        Compute the probability density function for a feature vector in a given class.

        Parameters:
        - _class: The class label.
        - x (numpy.ndarray): A single feature vector.

        Returns:
        - float: Probability density at P(x | class).
        """
        mean = self.class_means[_class]
        variance = self.class_variances[_class]
        numerator = np.exp(-(x - mean)**2 / (2*variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator #pdf at P(x | class)

    def plot(self, X, y, h=0.1):
        """
        Generate a decision boundary plot for the classifier.

        Parameters:
        - X (numpy.ndarray): The feature matrix.
        - y (numpy.ndarray): The array of class labels.
        - h (float): Step size in the mesh grid (default: 0.1).
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        unique_classes = np.unique(y)
        for c in unique_classes:
            class_data = X[y == c]
            plt.scatter(class_data[:, 0], class_data[:, 1], label=f'Class {c}')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Gaussian Naive Bayes Decision Boundary")
        plt.legend(loc='best')
        plt.show()

if __name__ == "__main__":
    n_samples = 1000
    X, y = test_data(n_samples)
    naive_model = GaussianBayesClassifer()
    naive_model.fit(X, y)
    naive_model.plot(X, y)


