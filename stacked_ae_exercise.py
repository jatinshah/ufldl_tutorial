import load_MNIST
import sparse_autoencoder
import scipy.optimize
import softmax
import stacked_autoencoder
import numpy as np

##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

input_size = 28 * 28
num_classes = 10
hidden_size_L1 = 200  # Layer 1 Hidden Size
hidden_size_L2 = 200  # Layer 2 Hidden Size
sparsity_param = 0.1  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term

##======================================================================
## STEP 1: Load data from the MNIST database
#
#  This loads our training data from the MNIST database files.

train_images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
train_labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')


##======================================================================
## STEP 2: Train the first sparse autoencoder
#  This trains the first sparse autoencoder on the unlabelled STL training
#  images.
#  If you've correctly implemented sparseAutoencoderCost.m, you don't need
#  to change anything here.

#  Randomly initialize the parameters
sae1_theta = sparse_autoencoder.initialize(hidden_size_L1, input_size)

J = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, input_size, hidden_size_L1,
                                                         lambda_, sparsity_param,
                                                         beta, train_images)
options_ = {'maxiter': 400, 'disp': True}

result = scipy.optimize.minimize(J, sae1_theta, method='L-BFGS-B', jac=True, options=options_)
sae1_opt_theta = result.x

print result

##======================================================================
## STEP 3: Train the second sparse autoencoder
#  This trains the second sparse autoencoder on the first autoencoder
#  featurse.
#  If you've correctly implemented sparseAutoencoderCost.m, you don't need
#  to change anything here.

sae1_features = sparse_autoencoder.sparse_autoencoder(sae1_opt_theta, hidden_size_L1,
                                                      input_size, train_images)

#  Randomly initialize the parameters
sae2_theta = sparse_autoencoder.initialize(hidden_size_L2, hidden_size_L1)

J = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, hidden_size_L1, hidden_size_L2,
                                                         lambda_, sparsity_param,
                                                         beta, sae1_features)

options_ = {'maxiter': 400, 'disp': True}

result = scipy.optimize.minimize(J, sae2_theta, method='L-BFGS-B', jac=True, options=options_)
sae2_opt_theta = result.x

print result


##======================================================================
## STEP 4: Train the softmax classifier
#  This trains the sparse autoencoder on the second autoencoder features.
#  If you've correctly implemented softmaxCost.m, you don't need
#  to change anything here.

sae2_features = sparse_autoencoder.sparse_autoencoder(sae2_opt_theta, hidden_size_L2,
                                                      hidden_size_L1, sae1_features)

options_ = {'maxiter': 400, 'disp': True}

softmax_theta, softmax_input_size, softmax_num_classes = softmax.softmax_train(hidden_size_L2, num_classes,
                                                                               lambda_, sae2_features,
                                                                               train_labels, options_)

##======================================================================
## STEP 5: Finetune softmax model

# Implement the stacked_autoencoder_cost to give the combined cost of the whole model
# then run this cell.


# Initialize the stack using the parameters learned
stack = [dict() for i in range(2)]
stack[0]['w'] = sae1_opt_theta[0:hidden_size_L1 * input_size].reshape(hidden_size_L1, input_size)
stack[0]['b'] = sae1_opt_theta[2 * hidden_size_L1 * input_size:2 * hidden_size_L1 * input_size + hidden_size_L1]
stack[1]['w'] = sae2_opt_theta[0:hidden_size_L1 * hidden_size_L2].reshape(hidden_size_L2, hidden_size_L1)
stack[1]['b'] = sae2_opt_theta[2 * hidden_size_L1 * hidden_size_L2:2 * hidden_size_L1 * hidden_size_L2 + hidden_size_L2]

# Initialize the parameters for the deep model
(stack_params, net_config) = stacked_autoencoder.stack2params(stack)

stacked_autoencoder_theta = np.concatenate((softmax_theta.flatten(), stack_params))

J = lambda x: stacked_autoencoder.stacked_autoencoder_cost(x, input_size, hidden_size_L2,
                                                           num_classes, net_config, lambda_,
                                                           train_images, train_labels)

options_ = {'maxiter': 400, 'disp': True}
result = scipy.optimize.minimize(J, stacked_autoencoder_theta, method='L-BFGS-B', jac=True, options=options_)
stacked_autoencoder_opt_theta = result.x

print result

##======================================================================
## STEP 6: Test

test_images = load_MNIST.load_MNIST_images('data/mnist/t10k-images.idx3-ubyte')
test_labels = load_MNIST.load_MNIST_labels('data/mnist/t10k-labels.idx1-ubyte')


# Two auto encoders without fine tuning
pred = stacked_autoencoder.stacked_autoencoder_predict(stacked_autoencoder_theta, input_size, hidden_size_L2,
                                                       num_classes, net_config, test_images)

print "Before fine-tuning accuracy: {0:.2f}%".format(100 * np.sum(pred == test_labels, dtype=np.float64) /
                                                     test_labels.shape[0])

# Two auto encoders with fine tuning
pred = stacked_autoencoder.stacked_autoencoder_predict(stacked_autoencoder_opt_theta, input_size, hidden_size_L2,
                                                       num_classes, net_config, test_images)

print "After fine-tuning accuracy: {0:.2f}%".format(100 * np.sum(pred == test_labels, dtype=np.float64) /
                                                    test_labels.shape[0])
