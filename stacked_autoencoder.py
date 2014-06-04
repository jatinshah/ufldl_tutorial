import numpy as np
import scipy.sparse
import softmax


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def stack2params(stack):
    """
    Converts a "stack" structure into a flattened parameter vector and also
    stores the network configuration. This is useful when working with
    optimization toolboxes such as minFunc.

    [params, netconfig] = stack2params(stack)

    stack - the stack structure, where stack{1}.w = weights of first layer
                                       stack{1}.b = weights of first layer
                                       stack{2}.w = weights of second layer
                                       stack{2}.b = weights of second layer
                                       ... etc.

    :param stack: the stack structure
    :return: params: flattened parameter vector
    :return: net_config: aux. variable with network structure
    """

    params = []
    for s in stack:
        params.append(s['w'].flatten())
        params.append(s['b'].flatten())
    params = np.concatenate(params)

    net_config = {}
    if len(stack) == 0:
        net_config['input_size'] = 0
        net_config['layer_sizes'] = []
    else:
        net_config['input_size'] = stack[0]['w'].shape[1]
        net_config['layer_sizes'] = []
        for s in stack:
            net_config['layer_sizes'].append(s['w'].shape[0])

    return params, net_config


def params2stack(params, net_config):
    """
    Converts a flattened parameter vector into a nice "stack" structure
    for us to work with. This is useful when you're building multilayer
    networks.
    stack = params2stack(params, netconfig)

    :param params: flattened parameter vector
    :param net_config: aux. variable containing network config.
    :return: stack structure (see above)

    """
    # Map the params (a vector into a stack of weights)
    depth = len(net_config['layer_sizes'])
    stack = [dict() for i in range(depth)]

    prev_layer_size = net_config['input_size']
    current_pos = 0

    for i in range(depth):
        # Extract weights
        wlen = prev_layer_size * net_config['layer_sizes'][i]
        stack[i]['w'] = params[current_pos:current_pos + wlen].reshape(net_config['layer_sizes'][i], prev_layer_size)
        current_pos = current_pos + wlen

        # Extract bias
        blen = net_config['layer_sizes'][i]
        stack[i]['b'] = params[current_pos:current_pos + blen]
        current_pos = current_pos + blen

        # Set previous layer size
        prev_layer_size = net_config['layer_sizes'][i]

    return stack


def stacked_autoencoder_cost(theta, input_size, hidden_size, num_classes,
                             net_config, lambda_, data, labels):
    """
    Takes a trained softmax_theta and a training data set with labels
    and returns cost and gradient using stacked autoencoder model.
    Used only for finetuning

    :param theta: trained weights from the autoencoder
    :param input_size: the number of input units
    :param hidden_size: the number of hidden units (at the layer before softmax)
    :param num_classes: number of categories
    :param net_config: network configuration of the stack
    :param lambda_: weight regularization penalty
    :param data: matrix containing data as columns. data[:,i-1] is i-th example
    :param labels: vector containing labels, labels[i-1] is the label for i-th example
    """

    ## Unroll softmax_theta parameter

    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size * num_classes].reshape(num_classes, hidden_size)

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size * num_classes:], net_config)

    m = data.shape[1]

    # Forward propagation
    a = [data]
    z = [np.array(0)]  # Dummy value

    for s in stack:
        z.append(s['w'].dot(a[-1]) + np.tile(s['b'], (m, 1)).transpose())
        a.append(sigmoid(z[-1]))

    # Softmax
    prod = softmax_theta.dot(a[-1])
    prod = prod - np.max(prod)
    prob = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())

    cost = (-1 / float(m)) * np.sum(indicator * np.log(prob)) + (lambda_ / 2) * np.sum(softmax_theta * softmax_theta)
    softmax_grad = (-1 / float(m)) * (indicator - prob).dot(a[-1].transpose()) + lambda_ * softmax_theta

    # Backprop
    # Compute partial of cost (J) w.r.t to outputs of last layer (before softmax)
    softmax_grad_a = softmax_theta.transpose().dot(indicator - prob)

    # Compute deltas
    delta = [-softmax_grad_a * sigmoid_prime(z[-1])]
    for i in reversed(range(len(stack))):
        d = stack[i]['w'].transpose().dot(delta[0]) * sigmoid_prime(z[i])
        delta.insert(0, d)

    # Compute gradients
    stack_grad = [dict() for i in range(len(stack))]
    for i in range(len(stack_grad)):
        stack_grad[i]['w'] = delta[i + 1].dot(a[i].transpose()) / m
        stack_grad[i]['b'] = np.sum(delta[i + 1], axis=1) / m

    grad_params, net_config = stack2params(stack_grad)
    grad = np.concatenate((softmax_grad.flatten(), grad_params))

    return cost, grad


def stacked_autoencoder_predict(theta, input_size, hidden_size, num_classes, net_config, data):
    """
    Takes a trained theta and a test data set,
    and returns the predicted labels for each example
    :param theta: trained weights from the autoencoder
    :param input_size: the number of input units
    :param hidden_size: the number of hidden units at the layer before softmax
    :param num_classes: the number of categories
    :param netconfig: network configuration of the stack
    :param data: the matrix containing the training data as columsn. data[:,i-1] is the i-th training example
    :return:

    Your code should produce the prediction matrix
    pred, where pred(i) is argmax_c P(y(c) | x(i)).
    """

    ## Unroll theta parameter
    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size * num_classes].reshape(num_classes, hidden_size)

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size * num_classes:], net_config)

    m = data.shape[1]

    # Compute predictions
    a = [data]
    z = [np.array(0)]  # Dummy value

    # Sparse Autoencoder Computation
    for s in stack:
        z.append(s['w'].dot(a[-1]) + np.tile(s['b'], (m, 1)).transpose())
        a.append(sigmoid(z[-1]))

    # Softmax
    pred = softmax.softmax_predict((softmax_theta, hidden_size, num_classes), a[-1])

    return pred