from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        i = 0
        for row in csv_reader:
            if not row:
                continue
            if i == 0:
                i += 1
                continue
            dataset.append(row)
    return dataset


def str_col_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_col_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup


def dataset_minmax(dataset):
    minmax = []
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])



def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = len(dataset) // n_folds
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    corr = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            corr += 1
    return corr*100/len(actual)


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation


def transfer(activation):
    return 1/(1+exp(-activation))


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


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


def back_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            back_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(">epoch={0}, LRate={1:.3f}, Error={2:.3f}".format(epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = []

    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)

    return predictions


filename = "seeds_dataset.csv"
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_col_to_float(dataset, i)

str_col_to_int(dataset, len(dataset[0])-1)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 6
l_rate = 0.5
n_epoch = 1000
n_hidden = 8

scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print("Scores: ", scores)
print("Mean Accuracy: ", sum(scores)/len(scores))
