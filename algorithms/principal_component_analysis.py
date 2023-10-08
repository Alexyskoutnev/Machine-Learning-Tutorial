import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Load the Iris dataset from a URL, preprocess it, and return the features (X),
    labels (y), and a dictionary mapping class names to integer labels.

    Returns:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Integer labels for each class.
    labels (dict): Dictionary mapping class names to integer labels.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    labels = dict()
    data = pd.read_csv(url, names=column_names)
    X, y = data.iloc[:,:-1].values, data['class'].values
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    labels = {str(idx+1): name  for idx, name in enumerate(label_encoder.classes_)} #Tranforms from output names to integer identifier (easier to deal with in data analysis)
    y = label_encoder.transform(y) + 1
    return X, y, labels

class PCA(object):
    def __init__(self, X, y):
        """
        Initialize the pca object with feature matrix X and integer class labels y.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.
        """
        self.X = X
        self.X_reduced = X
        self.y = y

    def solve(self, X, dim):
        """
        Perform Principal Component Analysis (PCA) on the input data.

        Args:
        dim (int): The number of dimensions for the reduced data.

        This method computes the PCA of the input data and reduces it to the specified number of dimensions.
        """
        num_samples = X.shape[0]
        mean = np.mean(X, axis=0)
        data = X - mean
        covar_mtx = 1 / num_samples * np.dot(data.T, data)
        eig_values, eig_vectors = np.linalg.eig(covar_mtx)
        idx = np.argsort(eig_values)[::-1]
        eig_vals_k, eig_vecs_k = eig_values[idx], eig_vectors[:, idx]
        eig_vec_k = eig_vectors[:, :dim]
        self.X_reduced = np.dot(data, eig_vec_k)

    def plot(self, labels):
        """
        Create a scatter plot to visualize the pca-reduced data.

        Args:
        labels (dict): Dictionary mapping class labels to class names.

        This function will display a scatter plot showing the pca-reduced data points
        in 2D space with different colors representing different classes.
        """
        plt.figure(figsize=(8, 6))
        class_labels = np.unique(self.y)
        colors = ['b', 'g', 'r']
        for i, label in enumerate(class_labels):
            plt.scatter(self.X_reduced[self.y == label, 0], self.X_reduced[self.y == label, 1], alpha=0.7, c=colors[i], label=labels[str(label)])
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('PCA Reduction')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    X, y, labels = load_data()
    new_dim = 2
    pca = PCA(X, y)
    pca.solve(X, new_dim)
    pca.plot(labels)