import random

from tools import *
from neuron import SimpleNeuralNetwork

train_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
train_y = [0, 1, 1, 0]

nn = SimpleNeuralNetwork(input_size=2, hidden_size=2)
nn.train(train_x, train_y)

nn.test(train_x, train_y)
