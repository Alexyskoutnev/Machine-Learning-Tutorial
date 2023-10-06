import numpy as np
import matplotlib.pyplot as plt

class MLE(object):
    """
    Maximum Likelihood Estimation (MLE) for Gaussian Distribution.

    This class provides methods for estimating the parameters of a Gaussian distribution
    using Maximum Likelihood Estimation (MLE).

    Attributes:
    - mu (float): The estimated mean (average) of the distribution.
    - sigma (float): The estimated standard deviation (spread) of the distribution.

    Methods:
    - fit(X): Fit the MLE model to the input data to estimate mu and sigma.
    - p(x): Calculate the probability density function (PDF) at a given value.

    Example usage:
    ```python
    mle_model = MLE()
    data = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
    mle_model.fit(data)
    pdf_value = mle_model.p(3.0)
    print(f"Estimated mu: {mle_model.mu}")
    print(f"Estimated sigma: {mle_model.sigma}")
    print(f"PDF at x=3.0: {pdf_value}")
    ```
    """
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def fit(self, X):
        """
        Fit the MLE model to the input data to estimate mu and sigma.

        Args:
        - X (numpy.ndarray): Input data as a 1D numpy array.

        Returns:
        None
        """
        num_samples = X.shape[0]
        self.mu = np.sum(X) / num_samples # dloss / dmu = sum(X) / n
        self.sigma = np.sqrt(np.sum((X - self.mu)**2) / num_samples) #dloss / dsigma = sqrt(sum(X - u)**2 / n)

    def p(self, x):
        """Calculate the probability density function (PDF) of a value.

        Args:
            x (float): The value for which to calculate the PDF.

        Returns:
            float: The PDF value at the given value `x`.
        """
        return (1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - self.mu)**2 / (2 * self.sigma**2)))) #Normal Distribution Equation

    def plot(self, X):        
        """
        Plot the histogram of the data along with the estimated PDF based on MLE.

        Args:
        - X (numpy.ndarray): Input data as a 1D numpy array.

        Returns:
        None
        """
        plt.hist(X, bins=30, density=True, alpha=0.5, color='b', label='Data Histogram')
        x_range = np.linspace(min(X), max(X), 1000)
        pdf_values = [self.p(x) for x in x_range]
        plt.plot(x_range, pdf_values, 'r-', lw=2, label='Estimated PDF (MLE)')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.title('Maximum Likelihood Estimation')
        plt.legend()
        plt.show()

def load_data(num_samples, mu=0, std=1, noise=False):
    data_normal = np.random.normal(mu, std, num_samples)
    if noise:
        noise_values = np.random.normal(0, 0.1, num_samples)
        data_normal += noise_values
    return data_normal

if __name__ == "__main__":
    num_samples = 10000
    mu = np.random.randint(10)
    std = np.random.randint(5)
    X = load_data(num_samples, mu, std, noise=True)
    mle_model = MLE()
    mle_model.fit(X)
    mle_model.plot(X)