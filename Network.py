import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


class Network:
    def __init__(
        self, data_location: str, classes: int, layers: list, input_size: int = -1
    ):
        self.data = np.array(pd.read_csv(data_location))

        input_size = self.data.shape[1] - 1 if input_size == -1 else input_size

        # Initialize variables
        self.m, self.n = self.data.shape

        # Initializes weights, biases, and change values for each layer
        self.weights_bias = {
            f"{i}": {
                "w": np.random.rand(layers[i + 1], layers[i]) - 0.5,
                "b": np.random.rand(layers[i + 1], 1) - 0.5,
                "dw": 0,
                "db": 0,
            }
            for i in range(len(layers))
            if i != len(layers) - 1
        }

        self.layers = len(layers) - 1

        self.W1 = np.random.rand(classes, input_size) - 0.5
        self.b1 = np.random.rand(classes, 1) - 0.5
        self.W2 = np.random.rand(classes, classes) - 0.5
        self.b2 = np.random.rand(classes, 1) - 0.5
        self.dW1 = 0
        self.db1 = 0
        self.dW2 = 0
        self.db2 = 0

        np.random.shuffle(self.data)

        # Initialize training and testing data
        self.train_size = int(0.8 * self.m)
        self.test_size = int(0.2 * self.m)

        self.train_data = self.data[self.test_size : self.m].T
        self.test_data = self.data[0 : self.test_size].T

        self.train_labels = self.train_data[0]
        self.train_items = self.train_data[1 : self.n]
        self.train_items = self.train_items / 255

        self.test_labels = self.test_data[0]
        self.test_items = self.test_data[1 : self.n]
        self.test_items = self.test_items / 255

    def relu(self, Z):
        return np.maximum(Z, 0)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    ### 2-layer neural network ###
    def forward_prop(self):
        Z1 = self.W1.dot(self.train_items) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2):
        one_hot_Y = self.one_hot(self.train_labels)
        dZ2 = A2 - one_hot_Y
        self.dW2 = 1 / self.m * dZ2.dot(A1.T)
        self.db2 = 1 / self.m * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.relu_deriv(Z1)
        self.dW1 = 1 / self.m * dZ1.dot(self.train_items.T)
        self.db1 = 1 / self.m * np.sum(dZ1)

    def update_params(self, alpha):
        self.W1 = self.W1 - alpha * self.dW1
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2

    def gradient_descent(self, alpha=0.1, iterations=500):
        for i in range(iterations + 1):
            Z1, A1, Z2, A2 = self.forward_prop()
            self.backward_prop(Z1, A1, Z2, A2)
            self.update_params(alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = np.argmax(A2, 0)
                print(
                    "Accuracy: ",
                    round(
                        (
                            np.sum(predictions == self.train_labels)
                            / self.train_labels.size
                        )
                        * 100,
                        3,
                    ),
                )

    def make_predictions(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        predictions = np.argmax(A2, 0)
        return predictions

    def test_prediction(self, index=-1):
        index = random.randint(0, self.train_size) if index == -1 else index
        current_image = self.train_items[:, index, None]
        prediction = self.make_predictions(self.train_items[:, index, None])
        label = self.train_labels[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()
    ### 2-layer neural network ###

    ### Generalized Neural Network ####
    def forward_prop_g(self):
        Z = [None for _ in range(self.layers)]
        A = [None for _ in range(self.layers)]
        for tag in self.weights_bias:
            weights = self.weights_bias[tag]["w"]
            biases = self.weights_bias[tag]["b"]

            # First layer is dotted with training data
            if tag == "0":
                Z[0] = weights.dot(self.train_items) + biases
            # Every other layer is dotted with activated data
            else:
                Z[int(tag)] = weights.dot(A[int(tag) - 1]) + biases

            # Last layer gets softmax function
            if tag == str(self.layers):
                A[self.layers - 1] = self.softmax(Z[int(tag)])
            # Every other layer gets ReLU function
            else:
                A[int(tag)] = self.relu(Z[int(tag)])
        return Z, A

    def backward_prop_g(self, Z, A):
        one_hot_y = self.one_hot(self.train_labels)
        dZ = [None for _ in range(self.layers)]

        for i in range(self.layers - 1, -1, -1):
            # Last layer one hot
            if i == self.layers - 1:
                dZ[i] = A[i] - one_hot_y
            else:
                dZ[i] = self.weights_bias[str(i + 1)]["w"].T.dot(
                    dZ[i + 1]
                ) * self.relu_deriv(Z[i])

            # First layer use training
            if i == 0:
                self.weights_bias["0"]["dw"] = (
                    1 / self.m * dZ[0].dot(self.train_items.T)
                )
            else:
                self.weights_bias[str(i)]["dw"] = (
                    1 / self.m * dZ[i].dot(A[i].T)
                )

            self.weights_bias[str(i)]["db"] = 1 / self.m * np.sum(dZ[i])

    def update_params_g(self, alpha):
        for tag in self.weights_bias:
            self.weights_bias[tag]["w"] = (
                self.weights_bias[tag]["w"] - alpha * self.weights_bias[tag]["dw"]
            )
            self.weights_bias[tag]["b"] = (
                self.weights_bias[tag]["b"] - alpha * self.weights_bias[tag]["db"]
            )

    def gradient_descent_g(self, alpha=0.1, iterations=500):
        for i in range(iterations + 1):
            Z, A = self.forward_prop_g()
            self.backward_prop_g(Z, A)
            self.update_params_g(alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = np.argmax(A[self.layers - 1], 0)
                print(
                    "Accuracy: ",
                    round(
                        (
                            np.sum(predictions == self.train_labels)
                            / self.train_labels.size
                        )
                        * 100,
                        3,
                    ),
                )

    def make_predictions_g(self):
        _, A = self.forward_prop_g()
        predictions = np.argmax(A[self.layers - 1], 0)
        return predictions

    def test_prediction_g(self, index=-1):
        index = random.randint(0, self.train_size) if index == -1 else index
        current_image = self.train_items[:, index, None]
        prediction = self.make_predictions(self.train_items[:, index, None])
        label = self.train_labels[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()
    ### Generalized Neural Network ####
