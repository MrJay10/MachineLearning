from random import random
from random import seed
from random import randint
from math import exp, tanh


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for j in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for j in range(n_outputs)]
    network.append(output_layer)
    return network


"""
# Sample code to check the weights of neurons for each level of initialized network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)
"""


def activate(weights, inputs):
    activation = weights[-1]

    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]

    return activation


def transfer(activation):
    return 1 / (1 + exp(-activation))


def forward_propagate(network, row):
    inputs = row

    for layer in network:
        new_inputs = []

        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])

        inputs = new_inputs

    return inputs


def transfer_derivative(output):
    return output*(1.0 - output)


def backpropagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for layer in range(len(network)):
        inputs = row
        if layer != 0:
            inputs = [neuron['output'] for neuron in network[layer-1]]
        for neuron in network[layer]:
            for j in range(len(inputs)-1):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backpropagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(">epoch={0}, LRate={1:.3f}, Error={2:.3f}".format(epoch, l_rate, sum_error))
        

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


if __name__ == '__main__':
    """
        Sample Code to train the neural network with a toy dataset
    """ 
    seed(1)
    # Toy dataset, to teach neural network to classify numbers less than equal to 10 as 0 and more than that as 1
    # dataset = [[randint(0, 21)] for i in range(50)]
    # for i in dataset:
    #     if i[0] <= 10:
    #         i.append(0)
    #     else:
    #         i.append(1)

    # random dataset
    dataset = [[2.7810836,2.550537003,0],
            [1.465489372,2.362125076,0],
            [3.396561688,4.400293529,0],
            [1.38807019,1.850220317,0],
            [3.06407232,3.005305973,0],
            [7.627531214,2.759262235,1],
            [5.332441248,2.088626775,1],
            [6.922596716,1.77106367,1],
            [8.675418651,-0.242068655,1],
            [7.673756466,3.508563011,1]]

    n_inputs = len(dataset[0])-1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset) - train_size
    train(network, dataset[:train_size], 0.1, 1000, n_outputs)

    """
    After training on complete dataset
    network = [[{'output': 0.025761543688909608, 'weights': [-1.4807948750189694, 1.8288089753512722, 1.077298011296571], 'delta': -0.006557001176357187}, {'output': 0.5138642612642116, 'weights': [-0.08635125604930567, 0.10436069180059236, 0.3212958744053779], 'delta': -0.0008492881050457321}], [{'output': 0.2815095843107191, 'weights': [2.2903922854737377, 0.7887233511355132, -1.4297748858703263], 'delta': -0.056938674159179796}, {'output': 0.7411295683749782, 'weights': [-2.6330229241652905, 0.8357651039198697, 0.7150581830860514], 'delta': 0.04966598305613813}]]
    """


    for row in dataset[-test_size:]:
        prediction = predict(network, row)
        print("Expected: {0}, Got: {1}".format(row[-1], prediction))
