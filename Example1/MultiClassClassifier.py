import numpy as np

class MultiClassClassifier:
    def __init__(self, num_classes, scale):
        self.num_classes = num_classes
        self.weights = None
        self.scale = scale

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8
            X = (X - self.mean) / self.std # Scale normalize X
        self.weights = np.zeros((X.shape[1], self.num_classes))

        for _ in range(epochs):
            logits = np.dot(X, self.weights)
            probabilities = self.softmax(logits)
            gradients = -np.dot(X.T, (np.eye(self.num_classes)[y] - probabilities)) / len(y)
            self.weights -= learning_rate * gradients

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        if self.scale:
            X_bias = (X_bias - self.mean) / self.std
        logits = np.dot(X_bias, self.weights)
        return np.argmax(logits, axis=1)