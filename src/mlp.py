import torch
import torch.nn.functional as F
import numpy as np

class MLP():
    """Implements a basic multi-layer perceptron for the iris dataset."""

    def __init__(self):
        """Constructor

        Initializes weights using He / Kaiming.  Biases set to zero.
        Inputs, hidden nodes, and outputs hard-coded to 4, 5, and 3
        respectively.
        """

        # Use hard-coded dimensions for now.
        self.inputs = 4
        self.outputs = 3
        self.hidden = 5

        # Initialize weights using He / Kaiming.
        w1 = torch.cat([
            torch.randn(self.hidden, self.inputs)*np.sqrt(2/self.inputs),
            torch.zeros(self.hidden, 1)], dim=1).double()
        w2 = torch.cat([
            torch.randn(self.outputs, self.hidden)*np.sqrt(2/self.hidden),
            torch.zeros(self.outputs, 1)], dim=1).double()

        self.weights = [w1, w2]

    def fit(self, X_train, y_train, epochs=30, lr=1):
        """Trains the MLP.

        Parameters:
            X_train - Torch tensor of iris samples (n, 4).
            y_train - Torch tensor of corresponding labels (n).
            epochs - Number of epochs.  Defaults to 30.
            lr - Learning rate, defaults to 1.
        """

        n = X_train.shape[0]
        # Append row of ones to the input to simplify bias addition.
        X = torch.cat([X_train.T, torch.ones(1, n)])
        # One-hot encode the labels.
        y = F.one_hot(y_train).T

        for i in range(epochs):
            w1 = self.weights[0]
            w2 = self.weights[1]

            # Call forward propagation to get the values needed to
            # calculate the gradients.
            z1, h1, z2, y_hat = self.forward(X)
            # Calculate the gradients with back propagation.
            g1, g2 = self.back(X, y, z1, h1, y_hat)

            print((-torch.sum(y*torch.log(y_hat))/n).item())

            # Update the weights.
            self.weights[0] = w1 - lr*g1
            self.weights[1] = w2 - lr*g2

    def forward(self, X):
        """Performs forward propagation.

        Parameters:
            X - Torch tensor of iris samples in column vector format with
                1's concatenated as the last feature. (5, n)
        Returns:
            z1 - The pre-ReLU values of the hidden layer. (5, n)
            h1 - The values of the hidden layer. (5, n)
            z2 - The pre-softmax values of the output layer. (3, n)
            y_hat - The values of the output layer. (3, n)
        """
        n = X.shape[1]
        w1 = self.weights[0]
        w2 = self.weights[1]

        # Calculate the hidden layer outputs.  Note that we add a row of
        # ones to simplify the bias addition similar to what we do to the
        # inputs.
        z1 = torch.matmul(w1, X)
        h1 = torch.cat([torch.relu(z1), torch.ones(1, n)])

        # Calculate the outputs.
        z2 = torch.matmul(w2, h1)
        y_hat = torch.softmax(z2, dim=0)

        return z1, h1, z2, y_hat

    def back(self, X, y, z1, h1, y_hat):
        """Performs back propagation.

        Parameters:
            X - Torch tensor of iris samples in column vector format with
                1's concatenated as the last feature. (5, n)
            y - One-hot encoded array of labels. (3, n)
            z1 - The pre-ReLU values of the hidden layer. (5, n)
            h1 - The values of the hidden layer. (5, n)
            y_hat - The values of the output layer. (3, n)
        Returns:
            g1 - The layer 1 gradients. (5, 5)
            g2 - The layer 2 gradients. (3, 6)
        """
        n = X.shape[1]
        w1 = self.weights[0]
        w2 = self.weights[1]

        # Calculate the error.
        e = y_hat - y
        # Calculate the layer 2 gradients.
        g2 = torch.matmul(e, torch.transpose(h1, 0, 1))/n
        # Calculate the layer 1 gradients.
        g1 = torch.matmul(torch.matmul(w2.T[:-1], e)*(z1 > 0), X.T)/n

        return g1, g2

    def predict(self, X_test):
        """Performs inference on a set of samples.

        Parameters:
            X_test - Torch tensor of the samples to predict. (n, 4)
        Returns:
            y_hat - The predicted classes. (n)
        """
        n = X_test.shape[0]
        X = torch.cat([X_test.T, torch.ones(1, n)])
        z1, h1, z2, y_hat = self.forward(X)
        return torch.argmax(y_hat, dim=0)

