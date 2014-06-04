import load_MNIST
import numpy as np
import softmax
import gradient

##======================================================================
## STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input.
#  We also initialise some parameters used for tuning the model.

# Size of input vector (MNIST images are 28x28)
input_size = 28 * 28
# Number of classes (MNIST images fall into 10 classes)
num_classes = 10
# Weight decay parameter
lambda_ = 1e-4
# Debug
debug = False

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels,
#  the input data is the images, and
#  the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as
# train-images.idx3-ubyte / train-labels.idx1-ubyte

images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')

if debug:
    input_size = 8 * 8
    input_data = np.random.randn(input_size, 100)
    labels = np.random.randint(num_classes, size=100)
else:
    input_size = 28 * 28
    input_data = images

# Randomly initialise theta
theta = 0.005 * np.random.randn(num_classes * input_size)


##======================================================================
## STEP 2: Implement softmaxCost
#
#  Implement softmaxCost in softmaxCost.m.

(cost, grad) = softmax.softmax_cost(theta, num_classes, input_size, lambda_, input_data, labels)

##======================================================================
## STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
#
if debug:
    J = lambda x: softmax.softmax_cost(x, num_classes, input_size, lambda_, input_data, labels)

    num_grad = gradient.compute_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    print num_grad, grad

    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print diff
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-7)\n\n"

##======================================================================
## STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct,
#  you can start training your softmax regression code using softmaxTrain
#  (which uses minFunc).

options_ = {'maxiter': 100, 'disp': True}
opt_theta, input_size, num_classes = softmax.softmax_train(input_size, num_classes,
                                                           lambda_, input_data, labels, options_)

##======================================================================
## STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmaxPredict
#  (in softmaxPredict.m), which should return predictions
#  given a softmax model and the input data.

test_images = load_MNIST.load_MNIST_images('data/mnist/t10k-images.idx3-ubyte')
test_labels = load_MNIST.load_MNIST_labels('data/mnist/t10k-labels.idx1-ubyte')
predictions = softmax.softmax_predict((opt_theta, input_size, num_classes), test_images)
print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == test_labels, dtype=np.float64) / test_labels.shape[0])