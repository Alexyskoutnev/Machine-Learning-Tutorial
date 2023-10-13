import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def load_data(num_samples=100, seed=1):
    np.random.seed(seed)
    X = np.random.rand(num_samples)
    noise = 0.1 * np.random.randn(num_samples)
    y = 2 * X + noise
    return X, y


class GradientBoostClassfier(object):
    pass

class GradientBoostRegressor(object):
    """
    A simple implementation of Gradient Boosting for regression tasks using decision trees.

    Parameters:
    - n_estimators (int): The number of weak learners (trees) in the ensemble. Default is 200.
    - learning_rate (float): The learning rate, or step size, to update the predictions. Default is 0.1.
    - max_depth (int): The maximum depth of the decision trees. Default is 3.

    Methods:
    - fit(X, y): Fit the gradient boosting model to the training data.
    - predict(X): Make predictions on new data.
    - plot(y, y_pred): Create a scatter plot of actual vs. predicted values.

    Attributes:
    - estimators: List of trained decision tree regressors used in the ensemble.
    """

    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3):
        """
        Initialize the GradientBoostRegressor.

        Args:
        - n_estimators (int): The number of weak learners (trees) in the ensemble.
        - learning_rate (float): The learning rate, or step size, to update the predictions.
        - max_depth (int): The maximum depth of the decision trees.
        """
        self.n_estimators = n_estimators 
        self.lr = learning_rate
        self.max_depth = 3
        self.estimators = []

    def fit(self, X, y):
        """
        Fit the gradient boosting model to the training data.

        Args:
        - X (array-like): The training input data.
        - y (array-like): The target values.
        """
        init_pred = np.mean(y) #First prediction of the mean of the y values
        predictions = np.full(len(y), init_pred)
        for _ in range(self.n_estimators):
            residuals = y - predictions #L2 loss gradient
            estimator = DecisionTreeRegressor(max_depth=self.max_depth) #use tree to find gradient based on the X values
            estimator.fit(X.reshape(-1, 1), residuals)
            predictions += self.lr * estimator.predict(X.reshape(-1, 1)) #use predicted gradient to update the prediction value
            self.estimators.append(estimator)

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
        - X (array-like): The input data for making predictions.

        Returns:
        - array: Predicted values.
        """
        predictions = np.full(X.shape[0], np.mean([estimator.predict(X.reshape(-1, 1)) for estimator in self.estimators]))
        for estimator in self.estimators:
            predictions += self.lr * estimator.predict(X.reshape(-1, 1))
        return predictions

    def plot(self, y, y_pred):
        """
        Create a scatter plot of actual vs. predicted values.

        Args:
        - y (array-like): Actual target values.
        - y_pred (array-like): Predicted values.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red', linewidth=2)
        plt.title("Boosted Tree Prediction")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    num_samples = 1000
    data = load_data(num_samples)
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=292)
    boosted_tree_regressor = GradientBoostRegressor()
    boosted_tree_regressor.fit(X_train, y_train)
    y_pred = boosted_tree_regressor.predict(X_test)
    boosted_tree_regressor.plot(y_test, y_pred)