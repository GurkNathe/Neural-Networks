# Neural-Networks ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This is based off of the presentation by Samson Zhang. You can find the code [here](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) and the video [here](https://www.youtube.com/watch?v=w8yWXqWQYmU).

This is an educational project that I am working on, while learning about Neural Networks and machine learning.

Currently there is a class called Network implemented in the Network.py file. The fully implemented version is a two-layer neural network implemented using the forward propagation, backward propagation method. 

There is a generalized neural network that is being worked on that is meant to allow any number of layers with any number of nodes. However, as of updating this, it is not complete for all cases, and has some issues. I've tested it with a two-layer neural network, similar to the hard-coded one, and it works approximately as well as the hard-coded one. You can add hidden layers, however they currently have to be the same size as the output layer, and they can be unstable in accuracy, to say the least.

My intent with this project is to create an object oriented neural network that is essentially plug-and-chug.

## Examples:

To use the neural network in its current form, you need a file with data, the number of classes/outputs from the network, and the layer structure.

```
network = Network("data/train.csv", classes=10, layers=[784, 10])
```

The gradient descent function is then called on the network to train it. This takes in two optional arguments: ```alpha``` (training rate), ```iterations``` (number of iterations). The default value for ```alpha``` is 0.1 and the default value for ```iterations``` is 500.

```
# Hard-coded version
network.gradient_descent(alpha=0.1, iterations=500)

# Generalized version
network.gradient_descent_g()
```

Currently, there is no output for the trained network, so it can either be used in a Jypiter notebook, or be trained every run when ran from the command line.

## Testing

To test the network, there are two functions implemented: ```test_prediction() & test_prediction_g()```. Each takes an optional argument ```index``` for the data point from the dataset to be tested against. If no index is given, a random datum is chosen to be tested against.