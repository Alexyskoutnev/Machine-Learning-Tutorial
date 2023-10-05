import matplotlib.pyplot as plt
import numpy as np

# Function to calculate coefficients (slope and intercept) of a linear regression model
def calc_coefficents(X, y):
    """
    Calculate the coefficients (slope and intercept) of a linear regression model.

    Parameters:
    - X: Array of independent variable values.
    - y: Array of dependent variable values.

    Returns:
    - b1: Slope coefficient.
    - b0: Intercept coefficient.
    """
    X_mean = np.mean(X)
    Y_mean = np.mean(y)
    b1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2) 
    b0 = Y_mean - b1 * X_mean
    return b0, b1

# Function to make predictions using a linear regression model
def predict(X, f):
    """
    Predict the dependent variable values using a linear regression model.

    Parameters:
    - X: Array of independent variable values.
    - f: lambda function of linear model

    Returns:
    - y_pred: Array of predicted dependent variable values.
    """
    return f(X)

# Function to generate random sample data for linear regression
def load_data(num_samples=100):
    """
    Generate random sample data for linear regression.

    Parameters:
    - num_samples: Number of data samples to generate.

    Returns:
    - X: Array of independent variable values.
    - Y: Array of dependent variable values.
    - coeff: Tuple of true coefficients (slope and intercept).
    """
    b0_true, b1_true = np.random.randint(10), np.random.randint(10)
    coeff = (b0_true, b1_true)
    X = np.random.rand(num_samples, 1)
    Y = b0_true + b1_true * X + np.random.rand(num_samples, 1) * 0.5
    return X, Y, coeff

# Function to plot data along with true and predicted regression lines
def plot(data, f_true=None, f_pred=None):
    """
    Plot the data along with true and predicted regression lines.

    Parameters:
    - data: Tuple containing X and Y data.
    - f_true: True regression line function.
    - f_pred: Predicted regression line function.

    Returns: None
    """
    X, y = data
    num_samples = len(X)
    min_X, max_X = min(X), max(X)
    _x = np.linspace(min_X, max_X, num_samples*10)
    y_true = predict(_x, f_true)
    y_pred = predict(_x, f_pred)
    print(f"MSE ERROR -> {mean_squared_error(y_true, y_pred):.2f}")
    plt.scatter(X, y)
    plt.plot(_x, y_true, label="True", color='red')
    plt.plot(_x, y_pred, label="Predicted", color='orange')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    plt.close()

def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between the true values and predicted values.

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - mse: Mean Squared Error (MSE).
    """    
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    return mse

if "__main__" == __name__:
    num_samples = 100
    X, Y, true_coeff = load_data(num_samples)
    data = (X, Y)
    predict_coeff = calc_coefficents(X, Y)
    f_true = lambda x, coeff=true_coeff : x * coeff[1]  + coeff[0]
    f_pred = lambda x, coeff=predict_coeff : x * coeff[1]  + coeff[0]
    plot(data, f_true, f_pred)