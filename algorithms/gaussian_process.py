import numpy as np
import matplotlib.pyplot as plt

class Kernel:

    rbf_kernel = "rbf"
    linear_kernel = "linear"
    polynomial_kernel = "polynomial"
    sigmoid_kernel = "sigmoid"
    hyperparameters = {}

    @classmethod
    def kernel(cls, kernel_type: str, **kwargs):
        cls.hyperparameters = kwargs
        if kernel_type == cls.rbf_kernel:
            return cls.rbf
        elif kernel_type == cls.linear_kernel:
            return cls.linear
        else:
            raise ValueError("Invalid kernel type")
    
    @classmethod
    def rbf(cls, x1, x2):
        a = cls.hyperparameters.get("a", 1.0)
        l = cls.hyperparameters.get("l", 1.0)
        sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return a * np.exp(-0.5 / l**2 * sq_dist)
    
    @classmethod
    def linear(cls, X, Y):
        return X @ Y.T    

class GaussianProcess:
    def __init__(self, kernel_type="rbf", noise=1e-8, *args, **kwargs):
        self.kernel = Kernel.kernel(kernel_type, **kwargs)
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel(X_train, X_train) + np.eye(X_train.shape[0]) * self.noise
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_test):
        K_test = self.kernel(self.X_train, X_test) # Finding covariance between training and test data
        mean = K_test.T @ np.linalg.solve(self.K, self.y_train) # K_X*_X @ K^-1 @ y finds the mean through out a targeted range
        cov = self.kernel(X_test, X_test) + self.noise * np.eye(len(X_test)) 
        cov -= K_test.T @ np.linalg.solve(self.K, K_test) # K_X*_X* - K_X*_X @ K^-1 @ K_X*_X finds the covariance throughout the targeted range
        return mean, cov

    @classmethod
    def rbf(x1, x2, a=1.0, l=1.0):
        sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return a * np.exp(-0.5 / l**2 * sq_dist)
    
    @classmethod
    def linear(cls, X, Y):
        return X @ Y.T   
    
    @staticmethod
    def plot(X_train, y_train, X_test, mean, cov):
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, c='red', marker='o', label='Training data')
        plt.plot(X_test, mean, color='blue', label='Mean prediction')
        plt.fill_between(X_test.flatten(), mean - 1.96 * np.sqrt(np.diag(cov)),
                        mean + 1.96 * np.sqrt(np.diag(cov)), color='lightblue', alpha=0.2, label='95% confidence interval')
        plt.title('Gaussian Process Regression')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.show()
    


if __name__ == "__main__":
    ## ==Hyperparameters==
    a = 2.0
    l = 1.0
    ## ===================
    X_train = np.sort(10 * np.random.rand(10, 1), axis=0)
    X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(len(X_train))
    gp = GaussianProcess(kernel_type="rbf", a=a, l=l)
    gp.fit(X_train, y_train)
    mean, cov = gp.predict(X_test)
    gp.plot(X_train, y_train, X_test, mean, cov)
