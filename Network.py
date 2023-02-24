import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

class Network():
    def __init__(self, data_location: str, classes: int, input_size: int=-1):
        self.data = np.array(pd.read_csv(data_location))
        
        input_size = self.data.shape[1] - 1 if input_size == -1 else input_size
        
        # Initialize variables
        self.m, self.n = self.data.shape
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
        self.train_size = int(0.8*self.m)
        self.test_size = int(0.2*self.m)

        self.train_data = self.data[self.test_size:self.m].T
        self.test_data = self.data[0:self.test_size].T

        self.train_labels = self.train_data[0]
        self.train_items = self.train_data[1:self.n]
        self.train_items = self.train_items / 255

        self.test_labels = self.test_data[0]
        self.test_items = self.test_data[1:self.n]
        self.test_items = self.test_items / 255
    
    def relu(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def forward_prop(self):
        Z1 = self.W1.dot(self.train_items) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def relu_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

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
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop()
            self.backward_prop(Z1, A1, Z2, A2)
            self.update_params(alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = np.argmax(A2, 0)
                print("Accuracy: ", round((np.sum(predictions == self.train_labels) / self.train_labels.size) * 100, 3))

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
        plt.imshow(current_image, interpolation='nearest')
        plt.show()