import numpy as np

# activation functions
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z) + 1e-8)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, std=0.0001):
        self.params = {'W1': std * np.random.randn(input_size, hidden_size), 
                       'b1': np.zeros(hidden_size),
                       'W2': std * np.random.randn(hidden_size, output_size), 
                       'b2': np.zeros(output_size)}

    def loss(self, train_x, train_y, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = train_x.shape[0]
        
        # forward pass
        z1 = np.dot(train_x, W1) + b1
        a1 = relu(z1)
        output = np.dot(a1,W2) + b2
        
        # softmax loss
        loss = 0
        output_max = np.max(output, axis=1, keepdims=True) 
        probs = softmax(output - output_max) 
        data_loss = np.sum(-np.log(probs[range(N), train_y])) / N  # cross-entropy loss
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) # L2-regularization
        loss = data_loss + reg_loss 
        
        # backward pass
        # compute gradients
        ds = probs.copy()
        ds[range(N), train_y] -= 1  
        ds /= N
        grads = {}
        grads['W2'] = np.dot(a1.T, ds)
        grads['b2'] = np.sum(ds, axis=0)
        dh = np.dot(ds, W2.T) 
        dh = (a1 > 0) * dh
        grads['W1'] = np.dot(train_x.T, dh) + reg * W1
        grads['b1'] = np.sum(dh, axis=0)
        grads['W2'] += reg * W2

        return loss, grads
             
    def predict(self, x):
        z1 = np.dot(x, self.params['W1']) + self.params['b1']
        a1 = relu(z1)
        output = np.dot(a1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(output, axis=1)            
        return y_pred     
    
    def train(self, train_x, train_y, val_x, val_y,
            lr=1e-3, lr_decay=0.9, reg=1e-6, 
            iteration=100, batch_size=200):
        N = train_x.shape[0]

        # SGD optimizer
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for i in range(iteration):
            # random minibatch
            indices = np.random.choice(np.arange(N), batch_size, replace = True)
            x_batch = train_x[indices,:]
            y_batch = train_y[indices]
            # loss and gradients
            loss, grads = self.loss(x_batch, y_batch, reg)
            train_loss_history.append(loss)
            # update parameters
            self.params['W1'] -= lr * grads['W1']
            self.params['b1'] -= lr * grads['b1']
            self.params['W2'] -= lr * grads['W2']
            self.params['b2'] -= lr * grads['b2']
            
            if i % 100 == 0:
                print('iteration {:d}/{:d}: loss {:.3f}'.format(i, iteration, loss))

            # check accuracy and loss
            loss, grads = self.loss(val_x, val_y, reg)
            val_loss_history.append(loss)
            train_acc = (self.predict(x_batch) == y_batch).mean()
            val_acc = (self.predict(val_x) == val_y).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # decay learning rate
            lr *= lr_decay

        return {'train_loss_history': train_loss_history,
                'train_acc_history': train_acc_history,
                'val_loss_history': val_loss_history,
                'val_acc_history': val_acc_history}  
    
    def save(self):
        np.savez('./trained_model.npz',W1=net.params['W1'],b1=net.params['b1'],
                 W2=net.params['W2'],b2=net.params['b2'])
