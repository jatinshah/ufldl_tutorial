import numpy as np


def load_MNIST_images(filename):
    """
    returns a 28x28x[number of MNIST images] matrix containing
    the raw MNIST images
    :param filename: input data file
    """
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_images, num_rows * num_cols)).transpose()
        images = images.astype(np.float64) / 255

        f.close()

        return images


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