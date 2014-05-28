import cPickle as pickle
import display_network
import numpy as np
import scipy.io
import cnn
import sparse_autoencoder
import sys

## CS294A/CS294W Convolutional Neural Networks Exercise

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  convolutional neural networks exercise. In this exercise, you will only
#  need to modify cnnConvolve.m and cnnPool.m. You will not need to modify
#  this file.

##======================================================================
## STEP 0: Initialization
#  Here we initialize some parameters used for the exercise.

image_dim = 64  # image dimension
image_channels = 3  # number of channels (rgb, so 3)

patch_dim = 8  # patch dimension
num_patches = 50000  # number of patches

visible_size = patch_dim * patch_dim * image_channels  # number of input units
output_size = visible_size  # number of output units
hidden_size = 400  # number of hidden units

epsilon = 0.1  # epsilon for ZCA whitening

pool_dim = 19  # dimension of pooling region

##======================================================================
## STEP 1: Train a sparse autoencoder (with a linear decoder) to learn
#  features from color patches. If you have completed the linear decoder
#  execise, use the features that you have obtained from that exercise,
#  loading them into optTheta. Recall that we have to keep around the
#  parameters used in whitening (i.e., the ZCA whitening matrix and the
#  meanPatch)
with open('stl10_features.pickle', 'r') as f:
    opt_theta = pickle.load(f)
    zca_white = pickle.load(f)
    patch_mean = pickle.load(f)

# Display and check to see that the features look good
W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
display_network.display_color_network(W.dot(zca_white).transpose(), 'zca_features_test.png')


##======================================================================
## STEP 2: Implement and test convolution and pooling
#  In this step, you will implement convolution and pooling, and test them
#  on a small part of the data set to ensure that you have implemented
#  these two functions correctly. In the next step, you will actually
#  convolve and pool the features with the STL10 images.

## STEP 2a: Implement convolution
#  Implement convolution in the function cnnConvolve in cnnConvolve.m

# Note that we have to preprocess the images in the exact same way
# we preprocessed the patches before we can obtain the feature activations.

stl_train = scipy.io.loadmat('data/stlTrainSubset.mat')
train_images = stl_train['trainImages']
train_labels = stl_train['trainLabels']
num_train_images = stl_train['numTrainImages'][0][0]

## Use only the first 8 images for testing
conv_images = train_images[:, :, :, 0:8]

convolved_features = cnn.cnn_convolve(patch_dim, hidden_size, conv_images,
                                      W, b, zca_white, patch_mean)

## STEP 2b: Checking your convolution
#  To ensure that you have convolved the features correctly, we have
#  provided some code to compare the results of your convolution with
#  activations from the sparse autoencoder

# For 1000 random points
for i in range(1000):
    feature_num = np.random.randint(0, hidden_size)
    image_num = np.random.randint(0, 8)
    image_row = np.random.randint(0, image_dim - patch_dim + 1)
    image_col = np.random.randint(0, image_dim - patch_dim + 1)

    patch = conv_images[image_row:image_row + patch_dim, image_col:image_col + patch_dim, :, image_num]

    ### REVIEW & FINALIZE - make it work with multiple channels
    patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
    patch = np.reshape(patch, (patch.size, 1))
    patch = patch - np.tile(patch_mean, (patch.shape[1], 1)).transpose()
    patch = zca_white.dot(patch)

    features = sparse_autoencoder.sparse_autoencoder(opt_theta, hidden_size, visible_size, patch)

    if abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
        print 'Convolved feature does not match activation from autoencoder'
        print 'Feature Number      :', feature_num
        print 'Image Number        :', image_num
        print 'Image Row           :', image_row
        print 'Image Column        :', image_col
        print 'Convolved feature   :', convolved_features[feature_num, image_num, image_row, image_col]
        print 'Sparse AE feature   :', features[feature_num, 0]
        sys.exit("Convolved feature does not match activation from autoencoder. Exiting...")

print 'Congratulations! Your convolution code passed the test.'