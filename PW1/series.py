from tools import normalize_data, denormalize_data
from neuron import SimpleNeuralNetwork


raw = [0.48, 4.30, 0.91, 4.85, 0.53, 4.51, 1.95, 5.88, 0.63, 5.79, 0.92, 5.18, 1.88, 4.84, 0.22]

train_x = [
    (0.48, 4.30, 0.91),
    (4.85, 0.53, 4.51),
    (1.95, 5.88, 0.63),
    (5.79, 0.92, 5.18)
]
expected = [4.85, 1.95, 5.79, 1.88]

min_val, max_val = min(raw), max(raw)

train_x_normalized = [normalize_data(x, min_val, max_val) for x in train_x]
expected_normalized = normalize_data(expected, min_val, max_val)

neuron = SimpleNeuralNetwork(3, 3, learning_rate=0.1)
neuron.train(train_x_normalized, expected_normalized)

print("Testing:")
predictions = [neuron.feedforward(x) for x in train_x_normalized]
predictions_denormalized = denormalize_data(predictions, min_val, max_val)


for i, inputs in enumerate(train_x):
    predicted = predictions_denormalized[i]
    actual = expected[i]
    error = abs((actual - predicted) / actual) * 100

    print(f"Input: {inputs} -> Predicted: {predicted:.2f} (Expected: {actual}) | Error: {error:.2f}%")

neuron.feedforward(raw)
