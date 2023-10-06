import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression(object):
    """
    Logistic Regression classifier.

    Parameters:
    - lr (float): Learning rate for gradient descent. Default is 0.001.
    - num_itr (int): Number of iterations for gradient descent. Default is 1000.

    Attributes:
    - lr (float): Learning rate for gradient descent.
    - num_itr (int): Number of iterations for gradient descent.
    - weights (array): Model coefficients.
    - bias (float): Model bias term.

    Methods:
    - fit(X, y): Fit the logistic regression model to the training data.
    - predict(X): Predict class labels for input data.

    Example usage:
    ```python
    model = LogisticRegression(lr=0.01, num_itr=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    def __init__(self, lr=0.01, num_itr=1000):
        """
        Initialize a logistic regression model.

        Parameters:
        - lr (float): Learning rate for gradient descent. Default is 0.001.
        - num_itr (int): Number of iterations for gradient descent. Default is 1000.
        """
        self.lr = lr
        self.num_itr = num_itr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X (array-like): Training data of shape (num_samples, num_features).
        - y (array-like): Target values of shape (num_samples,).

        Returns:
        None
        """
        num_samples, num_features = X.shape
        self.weights = np.ones(num_features)
        self.bias = 0
        for i in range(self.num_itr):
            x  = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(x)
            #Computing the gradients to find optimal coefficents for logit regression function
            dw = (1 / num_samples) * (np.dot(X.T, (y_pred - y))) # dloss/dw = X * (p(x) - Y)
            db = (1 / num_samples) * np.sum(y_pred - y) # dloss/db = -Y + p(x)
            #Perform gradient descent to update the model weights and bias
            self.bias -= db * self.lr
            self.weights -= dw * self.lr

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters:
        - X (array-like): Input data of shape (num_samples, num_features).

        Returns:
        - y (array): Predicted class labels (0 or 1) of shape (num_samples,).
        """
        x = np.dot(X, self.weights) + self.bias # logit(x) = b_0 + b_1 * x_1 + ... + b_n * x_n
        z = self.sigmoid(x)
        y = (z > 0.5).astype(int)
        return y

    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        Parameters:
        - z (array-like): Input values.

        Returns:
        - result (array): Sigmoid of input values.
        """

        return (1  / (1 + np.exp(-x)))

def plot(data, model):
    """
    Plot the decision boundary of a logistic regression model along with the data points.

    Parameters:
    - data (tuple): A tuple containing the input features and labels, where:
        - data[0] (numpy.ndarray): Input features of shape (num_samples, num_features).
        - data[1] (numpy.ndarray): Binary labels (0 or 1) of shape (num_samples,).
    - model (LogisticRegression): The trained logistic regression model.

    Returns:
    None

    This function generates a plot that displays the decision boundary of a logistic regression model
    and the data points. The regions where the model predicts Class 0 and Class 1 are color-coded, and
    a legend is provided to identify the classes. The plot also includes a colorbar to indicate class regions.
    """
    X, Y = data
    x_min, x_max = X.min() - 0.5, X.max() + 0.5
    y_min, y_max = Y.min() - 0.5, Y.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.colorbar(ticks=[0, 1], label='Class')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 0'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class 1')],
               title='Class Legend')
    plt.show()

def calc_error(model, X, y):
    """
    Calculate classification error and the number of correctly predicted instances.

    Parameters:
    - model: A trained classifier model (e.g., LogisticRegression).
    - X (array-like): Input data of shape (num_samples, num_features).
    - y (array-like): True target values of shape (num_samples,).

    Returns:
    - error_rate (float): Classification error rate, a value between 0 and 1 (lower is better).
    - correct_count (int): Number of correctly predicted instances.

    Example usage:
    ```python
    model = LogisticRegression(lr=0.01, num_itr=1000)
    model.fit(X_train, y_train)
    error, correct = calc_error(model, X_test, y_test)
    print("Error Rate:", error)
    print("Correct Predictions:", correct)
    ```
    """
    y_pred = model.predict(X)
    cnt = len(y)
    correct = sum(abs(y - y_pred))
    return 1 - (correct / cnt), correct

def load_data(num_samples=100, num_features=1, ceoff=None, seed=None):
    """
    Generate synthetic data for logistic regression.

    Parameters:
    - num_samples: Number of data samples to generate.
    - num_features: Number of features (independent variables).
    - coefficients: Coefficients for the logistic regression model. If None, random coefficients will be generated.
    - seed: Random seed for reproducibility.

    Returns:
    - X: Features matrix of shape (num_samples, num_features).
    - y: Binary labels vector (0 or 1) of shape (num_samples,).
    - coefficients: Coefficients used for data generation.
    """
    if seed is not None:
        seed = np.random.seed(seed)
    if ceoff is None:
        coeff = np.random.rand(num_features) * 10
    X = np.random.randn(num_samples, num_features) * 0.1
    logits = np.dot(X, coeff)
    probability = 1 / (1 + np.exp(-logits))
    Y = (probability > 0.5).astype(int)
    return X, Y

if __name__ == "__main__":
    num_samples = 100
    features = 2
    lr, num_itr = 0.01, 10000
    X, Y = load_data(num_samples, features)
    data = (X, Y)
    model = LogisticRegression(lr, num_itr)
    model.fit(X, Y)
    percentage_correct, num_correct = calc_error(model, X, Y)
    print(f"Model accurancy [{len(Y) - num_correct} / {len(Y)}] = {percentage_correct}%")
    plot(data, model)