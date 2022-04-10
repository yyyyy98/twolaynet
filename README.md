# twolaynet
Inspired by CS231n Assignment 1

In this project, we construct a two-layer neural network to distinguish between digits in MNIST image data set. We train the network with softmax loss and L2 regularization. To optimize the parameters, we use stochastic gradient decent. After tuning hyperparameters such as learning rates, hidden sizes and regularization strengths, we assess the performance of our neural network on test data.

The model.py file includes activation functions, loss and gradient calculation during backpropagation, learning rate descent strategy, L2 regularization and SGD optimizer. The param_selection.py file illustrates how to tune hyperparameters such as learning rates, hidden sizes and regularization strengths.
