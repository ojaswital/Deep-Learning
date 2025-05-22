import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=2000, print_cost=True):
        # Initialize hyperparameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.w = None  # weights
        self.b = None  # bias
        self.costs = []  # to track cost over iterations

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, dim):
        # Initialize weights and bias with zeros
        self.w = np.zeros((dim, 1))
        self.b = 0

    def propagate(self, X, Y):
        # Forward and backward propagation
        m = X.shape[1]

        # Forward
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # Backward
        dw = 1/m * np.dot(X, (A - Y).T)
        db = 1/m * np.sum(A - Y)

        grads = {"dw": dw, "db": db}
        return grads, cost

    def optimize(self, X, Y):
        # Optimize weights using gradient descent
        for i in range(self.num_iterations):
            grads, cost = self.propagate(X, Y)

            self.w -= self.learning_rate * grads["dw"]
            self.b -= self.learning_rate * grads["db"]

            if i % 100 == 0:
                self.costs.append(cost)
                if self.print_cost:
                    print(f"Cost after iteration {i}: {cost}")

        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def fit(self, X, Y):
        # Train the model
        self.initialize_parameters(X.shape[0])
        self.optimize(X, Y)

    def predict(self, X):
        # Predict binary labels
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        return (A > 0.5).astype(int)

    def evaluate(self, X, Y):
        # Evaluate accuracy
        Y_pred = self.predict(X)
        accuracy = 100 - np.mean(np.abs(Y_pred - Y)) * 100
        return accuracy
