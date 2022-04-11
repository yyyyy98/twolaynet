import numpy as np
import model
import os
import struct
import dataloader

# store best accuracy
best_acc = 0
best_stats = None
# hyperparameter lists
learning_rates = [1e-2, 5e-2, 1e-3, 5e-3]
reg_strengths = [1e-2, 5e-2, 1e-3, 5e-3]
hidden_sizes = [100, 300, 500]
results = {}
iters = 1000
input_size = 28 * 28
num_class = 10
batch_size = 200
data, label = dataloader.load_mnist('./mnist')
# split train into train and validation
train_x, train_y = data[0:50000, :], label[0:50000]
val_x, val_y = data[50000:, :], label[50000:]
# tune hyperparameters
for lr in learning_rates:
    for rs in reg_strengths:
        for hs in hidden_sizes:            
            net = model.TwoLayerNet(input_size, hs, num_class)
            stats = net.train(train_x, train_y, val_x, val_y, lr, 0.9, rs, iters, batch_size)
            train_y_pred = net.predict(train_x)
            train_acc = np.mean(train_y == train_y_pred)
            val_y_pred = net.predict(val_x)
            val_acc = np.mean(val_y == val_y_pred)
            results[(lr,rs,hs)] = (train_acc, val_acc)
            # update best results
            if best_acc < val_acc:
                best_stats = stats
                best_acc = val_acc
                best_net = net
for (lr,rs,hs) in sorted(results):
    (train_acc, val_acc) = results[(lr,rs,hs)]
    print('lr:%f,rs:%f,hs:%f,train_accuracy:%f,val_accuracy:%f' %(lr,rs,hs,train_acc,val_acc))
