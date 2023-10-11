import numpy as np
import cvxpy as cp

def check_psd(M):
    """
    Check if a given matrix is Positive Semi-Definite (PSD).

    Parameters:
        M (numpy.ndarray): The matrix to be checked.

    Returns:
        bool: True if the matrix is PSD, False otherwise.
    """
    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    # Check if all eigenvalues are non-negative
    is_psd = all(eigenvalues >= 0)
    if is_psd:
        return True
    else:
        return False

def make_psd(A, esp=1e-6):
    """
    Attempt to make a matrix PSD by modifying its eigenvalues.

    Parameters:
        A (numpy.ndarray): The input matrix.
        epsilon (float): A small positive constant to ensure non-negativity of eigenvalues.

    Returns:
        numpy.ndarray: The modified PSD matrix.
    """
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Check if any eigenvalues are negative
    if np.any(eigenvalues < 0):
        # Add a positive constant times the identity matrix to make eigenvalues non-negative
        A_psd = A + esp * np.identity(A.shape[0])
    else:
        # The matrix is already PSD
        A_psd = A

    return A_psd

class SVMLinear(object):
    pass

class SVM(object):
    """
    Support Vector Machine (SVM) classifier with optimization-based training.

    Parameters:
        C (float): The regularization parameter (default=1.0).
        kernel (str): The type of kernel to use ('linear' or 'rbf') (default='linear').
        max_itr (int): Maximum number of iterations for optimization (default=100).
        tol (float): Tolerance for convergence (default=1e-3).
    """
    def __init__(self, C=1.0, kernel='linear', max_itr=100, tol=1e-3) -> None:
        self.C = C
        self.kernel = kernel
        self.max_itr = max_itr
        self.tol = tol

    def fit(self, X, y):
        """
        Fit the SVM classifier to the training data.

        Parameters:
            X (numpy.ndarray): The training feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): The target labels of shape (n_samples,).

        Returns:
            None
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        # Create the kernel matrix
        self.K = self._kernel_matrix(X, X)

        # Define optimization variables
        alpha = cp.Variable(self.n_samples)

        # Define the objective function
        objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(cp.multiply(alpha, self.y), cp.psd_wrap(self.K)))

        # Define constraints
        constraints = [alpha >= 0, alpha <= self.C, cp.sum(cp.multiply(self.y, alpha)) == 0]

        # Create the optimization problem
        problem = cp.Problem(objective, constraints)

        # Solve the optimization problem
        problem.solve()

        # Extract the Lagrange multipliers (support vectors)
        self.support_vectors = X[alpha.value > 1e-3]
        self.support_vector_labels = y[alpha.value > 1e-3]
        self.alpha = alpha.value[alpha.value > 1e-3]

        # Compute the bias (intercept) term
        self.w = np.sum((self.alpha * self.support_vector_labels).reshape((1, len(self.alpha))).dot(self.support_vectors), axis=0)
        self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))

    def predict(self, X):
        """
        Predict the class labels for a given set of samples.

        Parameters:
            X (numpy.ndarray): The samples to predict, of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels (1 or -1).
        """
        if self.kernel == 'linear':
            decision_func = np.dot(X, self.w) + self.b
        elif self.kernel == 'rbf':
            decision_func = np.sum(self.alpha * self.y * self._rbf_kernel(X, self.X), axis=1) + self.b
        else:
            print(f"Kernel [{self.kernel}] has not been implemented")
        return np.sign(decision_func)

    def _rbf_kernel(self, x1, x2, gamma=0.1):
        """
        Compute the Radial Basis Function (RBF) kernel between two sets of samples.

        Parameters:
            x1 (numpy.ndarray): The first set of samples of shape (n_samples_1, n_features).
            x2 (numpy.ndarray): The second set of samples of shape (n_samples_2, n_features).
            gamma (float): The kernel parameter (default=0.1).

        Returns:
            numpy.ndarray: The kernel matrix of shape (n_samples_1, n_samples_2).
        """
        return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

    def _kernel_matrix(self, x1, x2):
        """
        Compute the kernel matrix for the given kernel type.

        Parameters:
            x1 (numpy.ndarray): The first set of samples of shape (n_samples_1, n_features).
            x2 (numpy.ndarray): The second set of samples of shape (n_samples_2, n_features).

        Returns:
            numpy.ndarray: The kernel matrix of shape (n_samples_1, n_samples_2).
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            K = np.zeros((x1.shape[0], x2.shape[0]))
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    K[i, j] = self._rbf_kernel(x1[i], x2[j]) #Getting rbf kernel -> K(xi, xj) = exp(-gamma*l2-norm(xi - xj))
            return K
        else:
            print("Kernel type not implemented")

def load_data(num_samples, features):
    """
    Generate a synthetic classification dataset and split it into training and testing sets.

    Parameters:
        num_samples (int): The total number of samples in the dataset.
        features (int): The number of features for each sample.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray: 
            X_train (numpy.ndarray): The training feature matrix of shape (n_samples_train, n_features).
            y_train (numpy.ndarray): The training target labels of shape (n_samples_train,).
            X_test (numpy.ndarray): The testing feature matrix of shape (n_samples_test, n_features).
            y_test (numpy.ndarray): The testing target labels of shape (n_samples_test,).
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=num_samples, n_features=features, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":
    num_samples, num_features = 100, 2
    X_train, y_train, X_test, y_test = load_data(num_samples, num_features)
    r_C = 1.0
    kernel = 'linear'
    svm = SVM(r_C, kernel)
    svm.fit(X_train, y_train)