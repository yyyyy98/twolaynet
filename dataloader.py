import numpy as np
import os
import struct

def load_mnist(path, kind='train'):
    if kind =='train':
        image_path = 'train-images-idx3-ubyte'
        label_path = 'train-labels-idx1-ubyte'
    else:
        image_path = 't10k-images-idx3-ubyte'
        label_path = 't10k-labels-idx1-ubyte'
    with open(os.path.join(path, label_path), 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(os.path.join(path, image_path), 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
