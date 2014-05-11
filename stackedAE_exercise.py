import load_MNIST
import numpy as np

##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

input_size = 28 * 28
num_classes = 10
hidden_size_L1 = 196  # Layer 1 Hidden Size
hidden_size_L2 = 196  # Layer 2 Hidden Size
sparsity_param = 0.1  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term

##======================================================================
## STEP 1: Load data from the MNIST database
#
#  This loads our training data from the MNIST database files.

images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')
