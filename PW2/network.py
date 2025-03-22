import numpy as np
import json

from functions import ActivationFunction


class NeuralNetwork:
    def __init__(self, input_size=None, hidden_sizes=None, output_size=None, activation='sigmoid', learning_rate=0.1, config_file=None):
        if config_file:
            self.load_model(config_file)
        else:
            self.learning_rate = learning_rate
            self.activation = ActivationFunction.sigmoid if activation == 'sigmoid' else ActivationFunction.relu
            self.activation_derivative = ActivationFunction.sigmoid_derivative if activation == 'sigmoid' else ActivationFunction.relu_derivative

            self.layers = [input_size] + hidden_sizes + [output_size]
            self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.1 for i in range(len(self.layers)-1)]
            self.biases = [np.random.randn(self.layers[i+1]) * 0.1 for i in range(len(self.layers)-1)]

    def forward(self, X):
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.activation(np.dot(X, w) + b)
            activations.append(X)
        return activations

    def backward(self, activations, y):
        deltas = [(activations[-1] - y) * self.activation_derivative(activations[-1])]

        for i in range(len(self.weights)-1, 0, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * self.activation_derivative(activations[i]))

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(activations, y)
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)[-1]

    def save_model(self, filename):
        model_data = {
            'config': {
                'input_size': self.layers[0],
                'hidden_sizes': self.layers[1:-1],
                'output_size': self.layers[-1],
                'activation': 'sigmoid' if self.activation == ActivationFunction.sigmoid else 'relu'
            },
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
            
        # Завантажуємо конфігурацію
        config = model_data['config']
        input_size = config['input_size']
        hidden_sizes = config['hidden_sizes']
        output_size = config['output_size']
        activation = config['activation']

        self.learning_rate = 0.1
        self.activation = ActivationFunction.sigmoid if activation == 'sigmoid' else ActivationFunction.relu
        self.activation_derivative = ActivationFunction.sigmoid_derivative if activation == 'sigmoid' else ActivationFunction.relu_derivative

        self.layers = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
