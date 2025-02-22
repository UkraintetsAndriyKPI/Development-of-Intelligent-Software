import random

from tools import *

class Neuron:
    def __init__(self, inputs_amount):
        self.weights = [random.uniform(-1, 1) for _ in range(inputs_amount)]
        self.bias = random.uniform(-1, 1)

    def train(self, training_data, epochs=1000, learning_rate=0.2):
        for _ in range(epochs):
            for inputs, expected in training_data:
                output = self.forward(inputs)
                err = expected - output
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * err * inputs[i]
                self.bias += learning_rate * err

    def forward(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i] * inputs[i]
        sum += self.bias
        return sigmoid_func(sum)



class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.3, max_iterations=10000, tolerance=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.w_input_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b_output = random.uniform(-1, 1)

    def feedforward(self, inputs):
        self.hidden_activations = [
            sigmoid_func(sum(inputs[k] * self.w_input_hidden[j][k] for k in range(self.input_size)) + self.b_hidden[j])
            for j in range(self.hidden_size)
        ]
        self.output = sigmoid_func(sum(self.hidden_activations[j] * self.w_hidden_output[j] for j in range(self.hidden_size))
                              + self.b_output)
        return self.output

    def train(self, train_x, train_y):
        for epoch in range(self.max_iterations):
            total_error = 0

            for i in range(len(train_x)):
                inputs = train_x[i]
                expected_output = train_y[i]

                output = self.feedforward(inputs)

                error = expected_output - output
                total_error += error ** 2

                delta_output = error * sigmoid_derivative(self.output)
                delta_hidden = [delta_output * self.w_hidden_output[j] * sigmoid_derivative(self.hidden_activations[j])
                                for j in range(self.hidden_size)]

                for j in range(self.hidden_size):
                    self.w_hidden_output[j] += self.learning_rate * delta_output * self.hidden_activations[j]
                    self.b_hidden[j] += self.learning_rate * delta_hidden[j]
                    for k in range(self.input_size):
                        self.w_input_hidden[j][k] += self.learning_rate * delta_hidden[j] * inputs[k]
                self.b_output += self.learning_rate * delta_output

            if total_error < self.tolerance:
                break

    def test(self, test_data, expected_results):
        print("Testing:")
        for i, inputs in enumerate(test_data):
            output = round(self.feedforward(inputs))
            print(f"Input: {inputs} -> Output: {output} (Expected: {expected_results[i]})")
