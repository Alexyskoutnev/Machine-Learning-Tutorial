import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_date(seed=1):
    """
    Load and split the MNIST dataset into training and testing sets.

    Parameters:
    - seed (int, optional): Random seed for data splitting. Default is 1.

    Returns:
    - X_train (array-like): Training data features.
    - X_test (array-like): Testing data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Testing data labels.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix heatmap for evaluating the performance of a classification model.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Displays:
    - A heatmap of the confusion matrix showing the true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

class XGBoost(object):
    """
    A custom class for building and using an XGBoost classifier for multi-class classification on the MNIST dataset.

    Methods:
    - __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3): Initialize the XGBoost classifier with hyperparameters.
    - fit(self, X, y): Train the XGBoost model using the provided data.
    - predict(self, X): Make predictions using the trained XGBoost model.

    Attributes:
    - n_estimators (int): The number of boosting rounds.
    - lr (float): The learning rate for the model.
    - max_depth (int): The maximum depth of individual decision trees.
    - model (XGBoost Booster): The trained XGBoost model.
    """

    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=2):
        """
        Initialize an XGBoost classifier with hyperparameters.

        Parameters:
        - n_estimators (int, optional): The number of boosting rounds. Default is 100.
        - learning_rate (float, optional): The learning rate for the model. Default is 0.1.
        - max_depth (int, optional): The maximum depth of individual decision trees. Default is 3.

        The method initializes the XGBoost classifier with the specified hyperparameters.

        Example:
        ```
        xgboost_model = XGBoost(n_estimators=200, learning_rate=0.01, max_depth=5)
        ```

        Attributes:
        - self.n_estimators (int): The number of boosting rounds.
        - self.lr (float): The learning rate for the model.
        - self.max_depth (int): The maximum depth of individual decision trees.
        - self.model (XGBoost Booster): The trained XGBoost model.

        Returns:
        None
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.model = None

    def predict(self, X):
        """
        Predict class labels using the trained XGBoost model.

        Parameters:
        - X (array-like): Input data for making predictions.

        Returns:
        - y_pred (array-like): Predicted class labels.

        The method takes input data and returns the predicted class labels using the trained XGBoost model.

        Example:
        ```
        y_pred = xgboost_model.predict(X_test)
        ```
        """
        dtest = xgb.DMatrix(X)
        y_prob = self.model.predict(dtest)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def fit(self, X, y):
        """
        Train the XGBoost model using the provided data.

        Parameters:
        - X (array-like): Training data features.
        - y (array-like): Training data labels.

        The method trains an XGBoost model with the given data and stores the trained model.

        Example:
        ```
        xgboost_model = XGBoost()
        xgboost_model.fit(X_train, y_train)
        ```

        Attributes:
        - self.model (XGBoost Booster): The trained XGBoost model.

        Hyperparameters:
        - self.n_estimators (int): The number of boosting rounds.
        - self.lr (float): The learning rate for the model.
        - self.max_depth (int): The maximum depth of individual decision trees.

        Returns:
        None
        """
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            'objective' : 'multi:softprob',
            'eval_metric' : 'mlogloss',
            'num_class' : len(np.unique(y)),
            'max_depth' : self.max_depth,
            'learning_rate' : self.lr,
            'n_estimators' : self.n_estimators,
        }
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_date()
    xgboost_model = XGBoost()
    xgboost_model.fit(X_train, y_train)
    y_pred = xgboost_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    print(f"Model accurancy: [{sum(y_test - y_pred) / len(y_test):.2f}]")

        