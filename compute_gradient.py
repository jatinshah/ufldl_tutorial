import numpy as np


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
        if (i % 100 == 0):
            print "Computing gradient for input:", i

    return gradient

