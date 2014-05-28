import numpy as np
import scipy.signal


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cnn_convolve(patch_dim, num_features, images, W, b, zca_white, patch_mean):
    """
    Returns the convolution of the features given by W and b with
    the given images
    :param patch_dim: patch (feature) dimension
    :param num_features: number of features
    :param images: large images to convolve with, matrix in the form
                   images(r, c, channel, image number)
    :param W: weights of the sparse autoencoder
    :param b: bias of the sparse autoencoder
    :param zca_white: zca whitening
    :param patch_mean: mean of the images
    :return:
    """

    num_images = images.shape[3]
    image_dim = images.shape[0]
    image_channels = images.shape[2]

    #  Instructions:
    #    Convolve every feature with every large image here to produce the
    #    numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1)
    #    matrix convolvedFeatures, such that
    #    convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
    #    value of the convolved featureNum feature for the imageNum image over
    #    the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
    #
    #  Expected running times:
    #    Convolving with 100 images should take less than 3 minutes
    #    Convolving with 5000 images should take around an hour
    #    (So to save time when testing, you should convolve with less images, as
    #    described earlier)

    convolved_features = np.zeros(shape=(num_features, num_images, image_dim - patch_dim + 1,
                                         image_dim - patch_dim + 1),
                                  dtype=np.float64)

    WT = W.dot(zca_white)
    bT = b - WT.dot(patch_mean)

    for i in range(num_images):
        for j in range(num_features):
            # convolution of image with feature matrix for each channel
            convolved_image = np.zeros(shape=(image_dim - patch_dim + 1, image_dim - patch_dim + 1),
                                       dtype=np.float64)

            for channel in range(image_channels):
                # Obtain the feature (patchDim x patchDim) needed during the convolution
                patch_size = patch_dim * patch_dim
                feature = WT[j, patch_size * channel:patch_size * (channel + 1)].reshape(patch_dim, patch_dim)

                # Flip the feature matrix because of the definition of convolution, as explained later
                feature = np.flipud(np.fliplr(feature))

                # Obtain the image
                im = images[:, :, channel, i]

                # Convolve "feature" with "im", adding the result to convolvedImage
                # be sure to do a 'valid' convolution
                convolved_image += scipy.signal.convolve2d(im, feature, mode='valid')

            # Subtract the bias unit (correcting for the mean subtraction as well)
            # Then, apply the sigmoid function to get the hidden activation
            convolved_image = sigmoid(convolved_image + bT[j])

            # The convolved feature is the sum of the convolved values for all channels
            convolved_features[j, i, :, :] = convolved_image

    return convolved_features


def cnn_pool(pool_dim, convolved_features):
    """
    Pools the given convolved features

    :param pool_dim: dimension of the pooling region
    :param convolved_features: convolved features to pool (as given by cnn_convolve)
                               convolved_features(feature_num, image_num, image_row, image_col)
    :return: pooled_features: matrix of pooled features in the form
                              pooledFeatures(featureNum, imageNum, poolRow, poolCol)
    """

    num_images = convolved_features.shape[1]
    num_features = convolved_features.shape[0]
    convolved_dim = convolved_features.shape[2]

    assert convolved_dim % pool_dim == 0, "Pooling dimension is not an exact multiple of convolved dimension"

    pool_size = convolved_dim / pool_dim
    pooled_features = np.zeros(shape=(num_features, num_images, pool_size, pool_size),
                               dtype=np.float64)

    for i in range(pool_size):
        for j in range(pool_size):
            pool = convolved_features[:, :, i * pool_dim:(i + 1) * pool_dim, j * pool_dim:(j + 1) * pool_dim]
            pooled_features[:, :, i, j] = np.mean(np.mean(pool, 2), 2)

    return pooled_features