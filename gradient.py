import numpy as np
import stacked_autoencoder


# this function accepts a 2D vector as input.
# Its outputs are:
#   value: h(x1, x2) = x1^2 + 3*x1*x2
#   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2
# Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
# that computeNumericalGradients will use only the first returned value of this function.
def simple_quadratic_function(x):
    value = x[0] ** 2 + 3 * x[0] * x[1]

    grad = np.zeros(shape=2, dtype=np.float32)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]

    return value, grad


# theta: a vector of parameters
# J: a function that outputs a real-number. Calling y = J(theta) will return the
# function value at theta.
def compute_gradient(J, theta):
    epsilon = 0.0001

    gradient = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        theta_epsilon_plus = np.array(theta, dtype=np.float64)
        theta_epsilon_plus[i] = theta[i] + epsilon
        theta_epsilon_minus = np.array(theta, dtype=np.float64)
        theta_epsilon_minus[i] = theta[i] - epsilon

        gradient[i] = (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * epsilon)
        if i % 100 == 0:
            print "Computing gradient for input:", i

    return gradient


# This code can be used to check your numerical gradient implementation
# in computeNumericalGradient.m
# It analytically evaluates the gradient of a very simple function called
# simpleQuadraticFunction (see below) and compares the result with your numerical
# solution. Your numerical gradient implementation is incorrect if
# your numerical solution deviates too much from the analytical solution.
def check_gradient():
    x = np.array([4, 10], dtype=np.float64)
    (value, grad) = simple_quadratic_function(x)

    num_grad = compute_gradient(simple_quadratic_function, x)
    print num_grad, grad
    print "The above two columns you get should be very similar.\n" \
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n"

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print diff
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n"


def check_stacked_autoencoder():
    """
    # Check the gradients for the stacked autoencoder
    #
    # In general, we recommend that the creation of such files for checking
    # gradients when you write new cost functions.
    #

    :return:
    """
    ## Setup random data / small model

    input_size = 64
    hidden_size_L1 = 36
    hidden_size_L2 = 25
    lambda_ = 0.01
    data = np.random.randn(input_size, 10)
    labels = np.random.randint(4, size=10)
    num_classes = 4

    stack = [dict() for i in range(2)]
    stack[0]['w'] = 0.1 * np.random.randn(hidden_size_L1, input_size)
    stack[0]['b'] = np.random.randn(hidden_size_L1)
    stack[1]['w'] = 0.1 * np.random.randn(hidden_size_L2, hidden_size_L1)
    stack[1]['b'] = np.random.randn(hidden_size_L2)
    softmax_theta = 0.005 * np.random.randn(hidden_size_L2 * num_classes)

    params, net_config = stacked_autoencoder.stack2params(stack)

    stacked_theta = np.concatenate((softmax_theta, params))

    cost, grad = stacked_autoencoder.stacked_autoencoder_cost(stacked_theta, input_size,
                                                              hidden_size_L2, num_classes,
                                                              net_config, lambda_, data, labels)

    # Check that the numerical and analytic gradients are the same
    J = lambda x: stacked_autoencoder.stacked_autoencoder_cost(x, input_size, hidden_size_L2,
                                                               num_classes, net_config, lambda_,
                                                               data, labels)
    num_grad = compute_gradient(J, stacked_theta)

    print num_grad, grad
    print "The above two columns you get should be very similar.\n" \
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n"

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print diff
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n"