# Neural-Networks ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This is based off of the presentation by Bot Academy. You can find the code [here](https://github.com/Bot-Academy/NeuralNetworkFromScratch/blob/master/nn.py) and the video [here](https://youtu.be/9RN2Wr8xvro).

This is an educational project that I am working on, while learning about Neural Networks and machine learning.

Currently there is a class called Network implemented in the Network.py file. The version implemented in the video by Bot Academy was a hard-coded neural network with a hidden layer of 20 nodes, input layer of 784 nodes, and output layer of 10 nodes. The network I have implemented is a generic network, where any layers sizes are accepted, and any number of layers are allowed (will change in the future to avoid errors).

My intent with this project is to create an object oriented neural network that is essentially plug-and-chug.

## Examples:

To use the neural network in its current form, you need training data, labels for the training data, and a list of the layers' node counts.

```
# Loading MNIST dataset from tensorflow
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
x_train = np.reshape(x_train, (x_train.shape[0],784), order='C')

network = Network(x_train, y_train, [784, 100, 10])
```

The gradient descent function is then called on the network to train it. This takes in two optional arguments: ```epochs``` (number of iterations), ```learning_rate``` (scaling factor while training). The default value for ```epochs``` is 3 and the default value for ```learning_rate``` is 0.1.

```
network.gradient_descent()
```

## Save & Loading

Currently, there is no output for the trained network, so it can either be used in a Jypiter notebook, or be trained every run when ran from the command line.

I am working on a way to save the model and load it, however it is still a work in progress, and doesn't work.

## Testing

To test the network, there is a prediction function implemented: ```predict()```. It takes an optional argument ```index``` for the data point from the dataset to be tested against. If no index is given, a random datum is chosen to be tested against.
