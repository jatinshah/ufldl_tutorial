import cPickle as pickle
import display_network
import numpy as np
import scipy.io
import cnn
import sparse_autoencoder
import sys
import time
import datetime
import softmax

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

## STEP 2c: Implement pooling
#  Implement pooling in the function cnnPool in cnnPool.m

# NOTE: Implement cnnPool in cnnPool.m first!

## STEP 2d: Checking your pooling
#  To ensure that you have implemented pooling, we will use your pooling
#  function to pool over a test matrix and check the results.
test_matrix = np.arange(64).reshape(8, 8)
expected_matrix = np.array([[np.mean(test_matrix[0:4, 0:4]), np.mean(test_matrix[0:4, 4:8])],
                            [np.mean(test_matrix[4:8, 0:4]), np.mean(test_matrix[4:8, 4:8])]])

test_matrix = np.reshape(test_matrix, (1, 1, 8, 8))

pooled_features = cnn.cnn_pool(4, test_matrix)

if not (pooled_features == expected_matrix).all():
    print "Pooling incorrect"
    print "Expected matrix"
    print expected_matrix
    print "Got"
    print pooled_features

print 'Congratulations! Your pooling code passed the test.'

##======================================================================
## STEP 3: Convolve and pool with the dataset
#  In this step, you will convolve each of the features you learned with
#  the full large images to obtain the convolved features. You will then
#  pool the convolved features to obtain the pooled features for
#  classification.
#
#  Because the convolved features matrix is very large, we will do the
#  convolution and pooling 50 features at a time to avoid running out of
#  memory. Reduce this number if necessary
step_size = 25
assert hidden_size % step_size == 0, "step_size should divide hidden_size"

stl_train = scipy.io.loadmat('data/stlTrainSubset.mat')
train_images = stl_train['trainImages']
train_labels = stl_train['trainLabels']
num_train_images = stl_train['numTrainImages'][0][0]

stl_test = scipy.io.loadmat('data/stlTestSubset.mat')
test_images = stl_test['testImages']
test_labels = stl_test['testLabels']
num_test_images = stl_test['numTestImages'][0][0]

pooled_features_train = np.zeros(shape=(hidden_size, num_train_images,
                                        np.floor((image_dim - patch_dim + 1) / pool_dim),
                                        np.floor((image_dim - patch_dim + 1) / pool_dim)),
                                 dtype=np.float64)
pooled_features_test = np.zeros(shape=(hidden_size, num_test_images,
                                       np.floor((image_dim - patch_dim + 1) / pool_dim),
                                       np.floor((image_dim - patch_dim + 1) / pool_dim)),
                                dtype=np.float64)

start_time = time.time()
for conv_part in range(hidden_size / step_size):
    features_start = conv_part * step_size
    features_end = (conv_part + 1) * step_size
    print "Step:", conv_part, "features", features_start, "to", features_end

    Wt = W[features_start:features_end, :]
    bt = b[features_start:features_end]

    print "Convolving & pooling train images"
    convolved_features = cnn.cnn_convolve(patch_dim, step_size, train_images,
                                          Wt, bt, zca_white, patch_mean)
    pooled_features = cnn.cnn_pool(pool_dim, convolved_features)
    pooled_features_train[features_start:features_end, :, :, :] = pooled_features

    print "Time elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))

    print "Convolving and pooling test images"
    convolved_features = cnn.cnn_convolve(patch_dim, step_size, test_images,
                                          Wt, bt, zca_white, patch_mean)
    pooled_features = cnn.cnn_pool(pool_dim, convolved_features)
    pooled_features_test[features_start:features_end, :, :, :] = pooled_features

    print "Time elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))

print('Saving pooled features...')
with open('cnn_pooled_features.pickle', 'wb') as f:
    pickle.dump(pooled_features_train, f)
    pickle.dump(pooled_features_test, f)

print "Saved"
print "Time elapsed:", str(datetime.timedelta(seconds=time.time() - start_time))

##======================================================================
## STEP 4: Use pooled features for classification
#  Now, you will use your pooled features to train a softmax classifier,
#  using softmaxTrain from the softmax exercise.
#  Training the softmax classifer for 1000 iterations should take less than
#  10 minutes.

# Load pooled features
with open('cnn_pooled_features.pickle', 'r') as f:
    pooled_features_train = pickle.load(f)
    pooled_features_test = pickle.load(f)

# Setup parameters for softmax
softmax_lambda = 1e-4
num_classes = 4

# Reshape the pooled_features to form an input vector for softmax
softmax_images = np.transpose(pooled_features_train, axes=[0, 2, 3, 1])
softmax_images = softmax_images.reshape((softmax_images.size / num_train_images, num_train_images))
softmax_labels = train_labels.flatten() - 1  # Ensure that labels are from 0..n-1 (for n classes)

options_ = {'maxiter': 1000, 'disp': True}
softmax_model = softmax.softmax_train(softmax_images.size / num_train_images, num_classes,
                                      softmax_lambda, softmax_images, softmax_labels, options_)

(softmax_opt_theta, softmax_input_size, softmax_num_classes) = softmax_model


##======================================================================
## STEP 5: Test classifer
#  Now you will test your trained classifer against the test images
softmax_images = np.transpose(pooled_features_test, axes=[0, 2, 3, 1])
softmax_images = softmax_images.reshape((softmax_images.size / num_test_images, num_test_images))
softmax_labels = test_labels.flatten() - 1

predictions = softmax.softmax_predict(softmax_model, softmax_images)
print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == softmax_labels, dtype=np.float64) / test_labels.shape[0])

# You should expect to get an accuracy of around 80% on the test images.
