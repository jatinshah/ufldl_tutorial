import numpy as np
import scipy.sparse
import scipy.optimize


def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):
    """

    :param theta:
    :param num_classes: the number of classes
    :param input_size: the size N of input vector
    :param lambda_: weight decay parameter
    :param data: the N x M input matrix, where each column corresponds
                 a single test set
    :param labels: an M x 1 matrix containing the labels for the input data
    """
    m = data.shape[1]
    theta = theta.reshape(num_classes, input_size)
    theta_data = theta.dot(data)
    theta_data = theta_data - np.max(theta_data)
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lambda_ / 2) * np.sum(theta * theta)

    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta

    return cost, grad.flatten()


def softmax_predict(model, data):
    # model - model trained using softmaxTrain
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    #
    # Your code should produce the prediction matrix
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).

    opt_theta, input_size, num_classes = model
    opt_theta = opt_theta.reshape(num_classes, input_size)

    prod = opt_theta.dot(data)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)

    return pred


def softmax_train(input_size, num_classes, lambda_, data, labels, options={'maxiter': 400, 'disp': True}):
    #softmaxTrain Train a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # input_size: the size of an input vector x^(i)
    # num_classes: the number of classes
    # lambda_: weight decay parameter
    # input_data: an N by M matrix containing the input data, such that
    #            inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)

    J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

    print result
    # Return optimum theta, input size & num classes
    opt_theta = result.x

    return opt_theta, input_size, num_classes

