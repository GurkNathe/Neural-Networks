# Neural-Networks ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This is based off of the presentation by Samson Zhang. You can find the code [here](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) and the video [here](https://www.youtube.com/watch?v=w8yWXqWQYmU).

This is an educational project that I am working on, while learning about Neural Networks and machine learning.

Currently there is a class called Network implemented in the Network.py file. It is a two-layer neural network implemented using the forward propagation, backward propagation method.

My intent with this project is to create an object oriented neural network that is essentially plug-and-chug.

Example:

To use the neural network in its current form, you need a file with data and the number of classes/outputs from the network.

```
network = Network("data/train.csv", classes=10)
```

The gradient descent function is then called on the network to train it. This takes in two optional arguments: training rate, number of iterations.

```
network.gradient_descent()
```

Currently, there is no output for the trained network, so it can either be used in a Jypiter notebook, or be trained every run when ran from the command line.