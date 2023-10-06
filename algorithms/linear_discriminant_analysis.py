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

class LDA(object):
    def __init__(self, X, y):
        """
        Initialize the LDA object with feature matrix X and integer class labels y.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.
        """
        self.X = X
        self.y = y

    def class_means(self, X, y):
        """
        Calculate the mean vector for each class in the dataset.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.

        Returns:
        class_mean_vector (dict): Dictionary with class labels as keys and
        corresponding class mean vectors as values.
        """
        class_labels = np.unique(y)
        class_mean_vector = {}
        for i in class_labels:
            class_mean_vector[i] = np.mean(X[y == i], axis=0)
        return class_mean_vector

    def class_within_scattter_mtx(self, X, y, class_mean_d):
        """
        Calculate the within-class scatter matrix (SW) for LDA.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.
        class_mean_d (dict): Dictionary with class labels as keys and
        corresponding class mean vectors as values.

        Returns:
        SW (numpy.ndarray): Within-class scatter matrix.
        """
        num_features = X.shape[1]
        SW = np.zeros((num_features, num_features))
        for label, mv in class_mean_d.items():
            class_scatter = np.zeros((num_features, num_features))
            for row in X[y == label]:
                row, mv = row.reshape(num_features, 1), mv.reshape(num_features, 1)
                class_scatter += (row - mv).dot((row - mv).T)
            SW += class_scatter
        return SW


    def class_between_scatter_mtx(self, X, y, class_mean_d):
        """
        Calculate the between-class scatter matrix (SB) for LDA.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.
        class_mean_d (dict): Dictionary with class labels as keys and
        corresponding class mean vectors as values.

        Returns:
        SB (numpy.ndarray): Between-class scatter matrix.
        """
        num_features = X.shape[1]
        overall_mean = np.mean(X, axis=0).reshape(num_features, 1)
        SB = np.zeros((num_features, num_features))
        for label, mv in class_mean_d.items():
            mv = mv.reshape(num_features, 1)
            SB += (mv - overall_mean).dot((mv - overall_mean).T)
        return SB

    def reduce(self, X, eigen, k=2):
        """
        Reduce the dimensionality of the dataset using LDA.

        Args:
        X (numpy.ndarray): Feature matrix.
        eigen (list): List of eigenvalues and eigenvectors.
        k (int): Number of dimensions to retain in the reduced space.

        Returns:
        Y (numpy.ndarray): Reduced feature matrix with k dimensions.
        """
        vec = []
        for i, (e_vec, e_val) in enumerate(eigen):
            if i >= k:
                break
            vec.append(e_vec)
        W = np.array(vec).T         
        Y = np.dot(X, W) # Y = X * W (Y is the new subspace for the dataset)
        return Y

    def plot(self, data, labels):
        """
        Create a scatter plot to visualize the LDA-reduced data.

        Args:
        data (numpy.ndarray): LDA-reduced feature matrix.
        labels (dict): Dictionary mapping class labels to class names.

        This function will display a scatter plot showing the LDA-reduced data points
        in 2D space with different colors representing different classes.
        """
        plt.figure(figsize=(8, 6))
        class_labels = np.unique(self.y)
        colors = ['b', 'g', 'r']
        for i, label in enumerate(class_labels):
            plt.scatter(data[self.y == label, 0], data[self.y == label, 1], alpha=0.7, c=colors[i], label=labels[str(label)])
        plt.xlabel('LDA 1')
        plt.ylabel('LDA 2')
        plt.title('LDA Reduction')
        plt.legend()
        plt.show()

    def solve(self, X, y):
        """
        Perform Linear Discriminant Analysis (LDA) on the dataset.

        Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Integer labels for each class.

        Returns:
        mu_dict (dict): Dictionary with class labels as keys and corresponding class
        mean vectors as values.
        SB (numpy.ndarray): Between-class scatter matrix.
        SW (numpy.ndarray): Within-class scatter matrix.
        eigen (list): List of eigenvalues and eigenvectors.
        """
        class_mu_vectors = self.class_means(X, y)
        SB = self.class_between_scatter_mtx(X, y, class_mu_vectors)
        SW = self.class_within_scattter_mtx(X, y, class_mu_vectors)
        eigen_val, eigen_vec = np.linalg.eig(np.linalg.inv(SW).dot(SB))
        eigen = [(eigen_vec[:, i], eigen_val[i]) for i in range(eigen_vec.shape[0])] 
        eigen.sort(key=lambda x: x[1], reverse=True)
        return class_mu_vectors, SB, SW, eigen

if __name__ == "__main__":
    X, y, labels = load_data()
    new_dim = 2
    lda = LDA(X, y)
    mu_dict, SB, SW, eigen = lda.solve(X, y)
    X_reduced = lda.reduce(X, eigen, new_dim)
    lda.plot(X_reduced, labels)