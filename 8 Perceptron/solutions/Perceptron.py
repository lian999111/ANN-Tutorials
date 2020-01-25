import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Perceptron:
    def __init__(self):
        self.weights = None
        self.current_training_error = None
        self.training_errors = None

    def train(self, X, Y, learning_rate, max_epochs, max_error, shuffle=True):
        self.weights = np.zeros(X.shape[1])  # Initialize weights
        self.current_training_error = self.count_errors(X, Y)
        self.training_errors = []

        # Run the perceptron algorithm
        for epoch_num in range(max_epochs):
            if shuffle:
                shuffled_indices = np.random.permutation(len(X))
                X = X[shuffled_indices]
                Y = Y[shuffled_indices]
            for x, y in zip(X, Y):
                self.training_step(X, Y, x, y, learning_rate)
                if self.current_training_error <= max_error:
                    print(f"Solution found at epoch {epoch_num}")
                    return

    def training_step(self, X, Y, x, y, learning_rate):
        y_predicted = 1 if np.dot(self.weights, x) > 0 else -1
        if y * y_predicted == -1:
            self.weights += 2 * learning_rate * y * x
            self.current_training_error = self.count_errors(X, Y)
        self.training_errors.append(self.current_training_error)

    def count_errors(self, X, Y):
        Y_predicted = np.where(np.dot(X, self.weights) > 0, 1, -1)
        return np.count_nonzero(Y*Y_predicted == -1)

    def plot_training_errors(self):
        fig, ax = plt.subplots()
        ax.plot(self.training_errors)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        return ax

    def plot_2D_decision_boundary(self, ax, X):
        x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 2)
        ax.plot(x1, -self.trained_weights()[0] / self.trained_weights()[2]
                - self.trained_weights()[1] / self.trained_weights()[2] * x1, 'k-')

    def trained_weights(self):
        return self.weights


class PocketPerceptron(Perceptron):
    def __init__(self):
        super().__init__()
        self.weights_in_pocket = None
        self.smallest_training_error = np.inf
        self.smallest_training_errors = []

    def training_step(self, X, Y, x, y, learning_rate):
        super().training_step(X, Y, x, y, learning_rate)
        if self.training_errors[-1] < self.smallest_training_error:
            self.weights_in_pocket = np.copy(self.weights)
            self.smallest_training_error = self.training_errors[-1]
        self.smallest_training_errors.append(self.smallest_training_error)

    def trained_weights(self):
        return self.weights_in_pocket

    def plot_training_errors(self):
        ax = super().plot_training_errors()
        ax.plot(self.smallest_training_errors, 'r', label='Best solution')
        ax.legend()
