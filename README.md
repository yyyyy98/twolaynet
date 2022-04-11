# Two-layer Neural Network Classifier
Inspired by CS231n Assignment 1

## Introduction
In this project, we construct a two-layer neural network to distinguish between digits in MNIST image data set. We train the network with softmax loss and L2 regularization. To optimize the parameters, we use stochastic gradient decent. After tuning hyperparameters such as learning rates, hidden sizes and regularization strengths, we assess the performance of our neural network on test data.

## Code
The `model.py` file includes activation functions, loss and gradient calculation during backpropagation, learning rate descent strategy, L2 regularization and SGD optimizer.

The `param_selection.py` file illustrates how to tune hyperparameters such as learning rates, hidden sizes and regularization strengths. 

In `test.py`, we assess the performance of the neural network on test data and visualize weights, loss and accuracy curves. 

The weights and biases of trained model were saved in `trained_model.npz`.



Run `load_model.py` to reproduce the testing accuracy in report.
