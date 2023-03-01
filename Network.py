import numpy as np
import random
import matplotlib.pyplot as plt

import timeit
import csv
from os import system, name, path

class Network():
    def __init__(self, data, labels, layers: list, testing_data=None, testing_labels=None):
        self.data = data
        # Onehot encode the labels
        self.labels = np.eye(10)[labels]

        self.testing_data = testing_data
        self.testing_labels = testing_labels

        self.layers = layers
        self.weights = [np.array([])] * (len(self.layers) - 1)
        self.biases = [np.array([])] * (len(self.layers) - 1)

        # Initialize weights and biases
        for i, item in enumerate(layers):
            if i < len(layers) - 1:
                self.weights[i] = np.random.rand(layers[i + 1], item) - 0.5
                self.biases[i] = np.random.rand(layers[i + 1], 1) - 0.5

    def gradient_descent(self, epochs: int=3, learning_rate:int=0.1):
        correct = 0
        for epoch in range(epochs):
            for image, label in zip(self.data, self.labels):
                image.shape += (1,)
                label.shape += (1,)
                
                # Forward propagation
                node_values = self.forward_prop(image, label)
                
                correct += int(np.argmax(node_values[len(node_values) - 1]) == np.argmax(label))
                
                self.backward_prop(node_values, label, learning_rate)
            # Show accuracy for this epoch
            print(f"Epoch: {epoch + 1}")
            print(f"Accuracy: {round((correct / self.data.shape[0]) * 100, 2)}%")
            correct = 0

    def forward_prop(self, image, label):
        values = [None] * (len(self.layers) - 1)
        values[0] = self.biases[0] + self.weights[0] @ np.array(image)
        values[0] = 1 / (1 + np.exp(-values[0]))

        for i in range(1, len(self.layers) - 1):
            values[i] = self.biases[i] + self.weights[i] @ np.array(values[i - 1])
            values[i] = 1 / (1 + np.exp(-values[i]))

        return values

    def backward_prop(self, values, label, learn_rate):
        delta = values[len(values) - 1] - label
        for i in range(len(self.weights) - 1, 0, -1):
            self.weights[i] += -learn_rate * delta @ values[i - 1].T
            self.biases[i] += -learn_rate * delta
            delta = self.weights[i].T @ delta * (values[i - 1] * (1 - values[i - 1]))

    def save_model(self):
        with open(
            path.join(
                path.dirname(__file__),
                f"models/model{int(timeit.default_timer())}.csv",
            ),
            "w",
            newline="",
        ) as myfile:
            wr = csv.writer(myfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            wr.writerow(self.layers)
            for layer in self.weights:
                for node in layer:
                    wr.writerow(node)
            for layer in self.biases:
                for node in layer:
                    wr.writerow(node)

    def load_model(self, file_path: str):
        data = []
        with open(file_path, "r") as model_data:
            values = csv.reader(model_data, delimiter=',')
            for row in values:
                data.append(row)

        # Get the layer sizes
        self.layers = [int(layer) for layer in data[:1][0]]

        row = 0
        for i in range(1, len(self.layers)):
            for node in range(self.layers[i]):
                self.weights[i - 1][node] = [*map(float, data[node + 1 + row])]
            row += self.layers[i]
        for i in range(1, len(self.layers)):
            for node in range(self.layers[i]):
                self.biases[i - 1][node] = [*map(float, data[node + 1 + row])]
            row += self.layers[i]

    def predict(self, index: int=-1):
        index = random.randint(0, self.data.shape[0]) if index == -1 else index
        image = self.data[index]
        plt.imshow(image.reshape(28, 28), cmap="Greys")
        
        image.shape += (1,)
        values = self.forward_prop(image, self.labels[index])
        plt.title(values[len(values) - 1].argmax())
        plt.show()
