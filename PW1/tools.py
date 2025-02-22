import math


def sigmoid_func(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def normalize_data(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]


def denormalize_data(data, min_val, max_val):
    return [x * (max_val - min_val) + min_val for x in data]
