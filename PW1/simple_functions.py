from neuron import Neuron

train_data = {
    'and': [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
    'or': [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
    'not': [([0], 1), ([1], 0)],
    # 'xor': [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
    'personal': [([0, 0, 0], 1), ([0, 1, 0], 1), ([1, 0, 0], 0), ([1, 1, 1], 1)]
}

for name, data in train_data.items():
    print("\n\tInput func/data:", name, data)

    inputs_amount = len(data[0][0])
    neuron = Neuron(inputs_amount)

    neuron.train(data)

    print(f"Testing '{name}':")
    for inputs, expected in data:
        output = neuron.forward(inputs)
        print(f"Input: {inputs} - Output: {round(output)} [Expected {expected}]")

    # print(neuron.weights)
