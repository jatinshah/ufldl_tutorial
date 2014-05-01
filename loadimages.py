import cPickle
import numpy as np


def unpickle(file_name):
    fo = open(file_name, 'rb')
    image_dict = cPickle.load(fo)
    fo.close()
    return image_dict


# Each column contains grayscale value for the image
# Squash data to [0.1, 0.9]
def normalize_data(images):
    # Subtract mean of each image from its individual values
    mean = images.mean(axis=0)
    images = images - mean

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd

    # Rescale from [-1,+1] to [0.1,0.9]
    images = (1 + images) * 0.4 + 0.1

    return images


# Convert RGB values to monochrome
def monochrome(r, g, b):
    return (0.2125 * r) + (0.7154 * g) + (0.0721 * b)


# Returns 10000 gray scale images for training from CIFAR-10 data
def load_images():
    image_size = 32
    num_images = 10000
    image_file = 'data/cifar10/data_batch_1'

    # Load Images & select first num_images images
    image_dict = unpickle(image_file)
    image_data = image_dict['data'][0:num_images]

    # Convert to grayscale & normalize
    red_data = image_data[:, 0:image_size * image_size]
    green_data = image_data[:, image_size * image_size:2 * image_size * image_size]
    blue_data = image_data[:, 2 * image_size * image_size:3 * image_size * image_size]

    grayscale_data = monochrome(red_data, green_data, blue_data)
    grayscale_data = normalize_data(grayscale_data.transpose())

    return normalize_data(grayscale_data)