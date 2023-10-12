import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def load_data(num_samples, features):    
    """
    Generate random data for a binary classification problem.

    Parameters:
    - num_samples (int): The number of data samples to generate.
    - features (int): The number of features for each sample.

    Returns:
    - X (torch.Tensor): The feature tensor of shape (num_samples, features).
    - y (torch.Tensor): The target tensor of shape (num_samples,), with binary labels (0 or 1).
    """
    input_size = features
    X = np.random.rand(num_samples, input_size)
    y = np.random.randint(2, size=num_samples)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

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
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

class NNClassifer(nn.Module):
    
    def __init__(self, input_size, output_size, layer_sizes=[32, 32, 32], activation_fn=nn.ReLU()):
        """
        Initialize a neural network for binary classification.

        Parameters:
        - input_size (int): The number of input features.
        - output_size (int): The number of output units (1 for binary classification).
        - layer_sizes (list): A list of hidden layer sizes.
        - activation_fn (nn.Module): The activation function to use in hidden layers.
        """
        super().__init__()
        self.relu = activation_fn
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, layer_sizes[i+1]))
                layers.append(activation_fn)
            else:
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i]))
                layers.append(activation_fn)
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        - x (torch.Tensor): The input feature tensor of shape (num_samples, input_size).

        Returns:
        - output (torch.Tensor): The output tensor of shape (num_samples, output_size).
        """
        x = self.network(x)
        return x

    def plot(self, X, y):
        """
        Plot the decision boundary of the binary classifier.

        Parameters:
        - X (torch.Tensor): The feature tensor of shape (num_samples, 2).
        - y (torch.Tensor): The target tensor of shape (num_samples,) with class labels.
        """

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        mesh_data = np.c_[xx.ravel(), yy.ravel()]
        
        with torch.no_grad():
            Z = self.network(torch.tensor(mesh_data, dtype=torch.float32)).numpy().reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=20)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

def train(network, X, y, epochs=10, lr=0.001, batch_size=64):
    """
    Train the binary classifier network.

    Parameters:
    - network (NNClassifer): The neural network to train.
    - X (torch.Tensor): The feature tensor of shape (num_samples, 2).
    - y (torch.Tensor): The target tensor of shape (num_samples,) with binary labels (0 or 1).
    - epochs (int): The number of training epochs.
    - lr (float): The learning rate for optimization.
    """
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
            total_samples = X.size(0)
            for i in range(0, total_samples, batch_size):
                # Get a mini-batch
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()  # Zero the gradient buffers
                outputs = network(batch_X)  # Forward pass
                loss = criterion(outputs, batch_y.view(-1, 1))  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    input_size, output_size = 2, 1
    epochs, lr = 1000, 0.01 #training hyperparameters for a trivial neural network (number of times to go over dataset and the gradient step for backprob)
    num_samples, feature = 1000, input_size
    network = NNClassifer(input_size, output_size)
    X, y = test_data(num_samples)
    train(network, X, y, epochs=epochs, lr=lr)
    x = torch.tensor([[1, 1]], dtype=torch.float32)
    y_pred = network(x)
    network.plot(X, y)

