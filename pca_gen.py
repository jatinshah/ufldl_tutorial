import sample_images
import random
import display_network
import numpy as np


##================================================================
## Step 0a: Load data
#  Here we provide the code to load natural image data into x.
#  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
#  the raw image data from the kth 12x12 image patch sampled.
#  You do not need to change the code below.

patches = sample_images.sample_images_raw()
num_samples = patches.shape[1]
random_sel = random.sample(range(num_samples), 400)
display_network.display_network(patches[:, random_sel], 'raw_pca.png')

##================================================================
## Step 0b: Zero-mean the data (by row)
#  You can make use of the mean and repmat/bsxfun functions.

# patches = patches - patches.mean(axis=0)
patch_mean = patches.mean(axis=1)
patches = patches - np.tile(patch_mean, (patches.shape[1], 1)).transpose()

##================================================================
## Step 1a: Implement PCA to obtain xRot
#  Implement PCA to obtain xRot, the matrix in which the data is expressed
#  with respect to the eigenbasis of sigma, which is the matrix U.

sigma = patches.dot(patches.transpose()) / patches.shape[1]
(u, s, v) = np.linalg.svd(sigma)

patches_rot = u.transpose().dot(patches)

##================================================================
## Step 2: Find k, the number of components to retain
#  Write code to determine k, the number of components to retain in order
#  to retain at least 99% of the variance.

k = 0
for k in range(s.shape[0]):
    if s[0:k].sum() / s.sum() >= 0.99:
        break
print 'Optimal k to retain 99% variance is:', k

##================================================================
## Step 3: Implement PCA with dimension reduction
#  Now that you have found k, you can reduce the dimension of the data by
#  discarding the remaining dimensions. In this way, you can represent the
#  data in k dimensions instead of the original 144, which will save you
#  computational time when running learning algorithms on the reduced
#  representation.
# 
#  Following the dimension reduction, invert the PCA transformation to produce 
#  the matrix xHat, the dimension-reduced data with respect to the original basis.
#  Visualise the data and compare it to the raw data. You will observe that
#  there is little loss due to throwing away the principal components that
#  correspond to dimensions with low variation.

patches_tilde = u[:, 0:k].transpose().dot(patches)
patches_hat = u.dot(np.resize(patches_tilde, patches.shape))

display_network.display_network(patches_hat[:, random_sel], 'pca_tilde.png')
display_network.display_network(patches[:, random_sel], 'pca.png')

##================================================================
## Step 4a: Implement PCA with whitening and regularisation
#  Implement PCA with whitening and regularisation to produce the matrix
#  xPCAWhite.

epsilon = 0.1
patches_pcawhite = np.diag(1 / (s + epsilon)).dot(patches_rot)


##================================================================
## Step 5: Implement ZCA whitening
#  Now implement ZCA whitening to produce the matrix xZCAWhite.
#  Visualise the data and compare it to the raw data. You should observe
#  that whitening results in, among other things, enhanced edges.

patches_zcawhite = u.dot(patches_pcawhite)
display_network.display_network(patches_zcawhite[:, random_sel], 'pca_zcawhite.png')
