import numpy as np
import model
import dataloader
import matplotlib.pyplot as plt

train_x, train_y = dataloader.load_mnist('./mnist')
test_x, test_y = dataloader.load_mnist('./mnist','t10k')
net = model.TwoLayerNet(28 * 28, 300, 10)
stats = net.train(train_x, train_y, test_x, test_y, 0.005, 0.95, 0.05, 1000, 200)

test_acc = (net.predict(test_x) == test_y).mean()
test_acc

# save model
# net.save()

# show loss and accuracy curves
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(stats['train_loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('training loss')

plt.subplot(1, 3, 2)
plt.plot(stats['val_loss_history'])
plt.xlabel('iteration')
plt.ylabel('testing loss')
plt.title('testing loss')

plt.subplot(1, 3, 3)
plt.plot(stats['val_acc_history'])
plt.xlabel('iteration')
plt.ylabel('testing accuracy')
plt.title('testing accuracy')
plt.show()

# show weights
def visualize_grid(net, weight):
    W1 = net.params[weight]
    W1 = W1.reshape(10, 10, 3, -1)

    padding = 1
    (H, W, C, N) = W1.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
        
    index = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if index < N:
                img = W1[:,:,:,index]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255 * (img - low) / (high - low)
                index += 1
            x0 += W + padding                
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    plt.imshow(grid.astype('uint8'))
    plt.title(weight)
    plt.show()
    
visualize_grid(net, 'W1')
visualize_grid(net, 'W2')
