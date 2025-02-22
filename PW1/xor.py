import random

from tools import *
from neuron import SimpleNeuralNetwork

# Данные для тренировки (XOR)
train_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
train_y = [0, 1, 1, 0]

# Создание и обучение нейросети
nn = SimpleNeuralNetwork(input_size=2, hidden_size=2)
nn.train(train_x, train_y)

# Тестирование
nn.test(train_x, train_y)
