import numpy as np


def load_MNIST_labels(filename):
    """
    returns a [number of MNIST images]x1 matrix containing
    the labels for the MNIST images

    :param filename: input file with labels
    """
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        labels = np.fromfile(f, dtype=np.ubyte)

        f.close()

        return labels


load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')