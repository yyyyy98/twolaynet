import numpy as np
import model
import dataloader
net = model.TwoLayerNet(784, 300, 10)
# load parameters of trained model
net.params['W1'] = np.load('./trained_model.npz')['W1']
net.params['b1'] = np.load('./trained_model.npz')['b1']
net.params['W2'] = np.load('./trained_model.npz')['W2']
net.params['b2'] = np.load('./trained_model.npz')['b2']

# compute testing accuracy
test_x, test_y = dataloader.load_mnist('./mnist', 't10k')
np.mean(test_y == net.predict(test_x))
