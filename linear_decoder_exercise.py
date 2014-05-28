import numpy as np
import sparse_autoencoder
import gradient
import scipy.io
import display_network
import scipy.optimize
import cPickle

##======================================================================
## STEP 0: Initialization
#  Here we initialize some parameters used for the exercise.

image_channels = 3  # number of channels (rgb, so 3)

patch_dim = 8  # patch dimension
num_patches = 100000  # number of patches

visible_size = patch_dim * patch_dim * image_channels  # number of input units
output_size = visible_size  # number of output units
hidden_size = 400  # number of hidden units

sparsity_param = 0.035  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 5  # weight of sparsity penalty term

epsilon = 0.1  # epsilon for ZCA whitening

##======================================================================
## STEP 1: Create and modify sparseAutoencoderLinearCost.m to use a linear decoder,
#          and check gradients
#  You should copy sparseAutoencoderCost.m from your earlier exercise
#  and rename it to sparseAutoencoderLinearCost.m.
#  Then you need to rename the function from sparseAutoencoderCost to
#  sparseAutoencoderLinearCost, and modify it so that the sparse autoencoder
#  uses a linear decoder instead. Once that is done, you should check
# your gradients to verify that they are correct.

# NOTE: Modify sparseAutoencoderCost first!

# To speed up gradient checking, we will use a reduced network and some
# dummy patches

debug_hidden_size = 5
debug_visible_size = 8
patches = np.random.rand(8, 10)

theta = sparse_autoencoder.initialize(debug_hidden_size, debug_visible_size)

cost, grad = sparse_autoencoder.sparse_autoencoder_linear_cost(theta, debug_visible_size, debug_hidden_size,
                                                               lambda_, sparsity_param, beta, patches)

# Check gradients
J = lambda x: sparse_autoencoder.sparse_autoencoder_linear_cost(x, debug_visible_size, debug_hidden_size,
                                                                lambda_, sparsity_param, beta, patches)
num_grad = gradient.compute_gradient(J, theta)

print grad, num_grad

# Compare numerically computed gradients with the ones obtained from backpropagation
diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
print diff
print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n"

##======================================================================
## STEP 2: Learn features on small patches
#  In this step, you will use your sparse autoencoder (which now uses a
#  linear decoder) to learn features on small patches sampled from related
#  images.

## STEP 2a: Load patches
#  In this step, we load 100k patches sampled from the STL10 dataset and
#  visualize them. Note that these patches have been scaled to [0,1]

patches = scipy.io.loadmat('data/stlSampledPatches.mat')['patches']

display_network.display_color_network(patches[:, 0:100], filename='patches_raw.png')


## STEP 2b: Apply preprocessing
#  In this sub-step, we preprocess the sampled patches, in particular,
#  ZCA whitening them.
#
#  In a later exercise on convolution and pooling, you will need to replicate
#  exactly the preprocessing steps you apply to these patches before
#  using the autoencoder to learn features on them. Hence, we will save the
#  ZCA whitening and mean image matrices together with the learned features
#  later on.

# Subtract mean patch (hence zeroing the mean of the patches)
patch_mean = np.mean(patches, 1)
patches = patches - np.tile(patch_mean, (patches.shape[1], 1)).transpose()

# Apply ZCA whitening
sigma = patches.dot(patches.transpose()) / patches.shape[1]
(u, s, v) = np.linalg.svd(sigma)
zca_white = u.dot(np.diag(1 / (s + epsilon))).dot(u.transpose())
patches_zca = zca_white.dot(patches)

display_network.display_color_network(patches_zca[:, 0:100], filename='patches_zca.png')

## STEP 2c: Learn features
#  You will now use your sparse autoencoder (with linear decoder) to learn
#  features on the preprocessed patches. This should take around 45 minutes.

theta = sparse_autoencoder.initialize(hidden_size, visible_size)

options_ = {'maxiter': 400, 'disp': True}

J = lambda x: sparse_autoencoder.sparse_autoencoder_linear_cost(x, visible_size, hidden_size,
                                                                lambda_, sparsity_param, beta, patches_zca)

result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x
print result

# Save the learned features and the preprocessing matrices for use in
# the later exercise on convolution and pooling
print('Saving learned features and preprocessing matrices...')
with open('stl10_features.pickle', 'wb') as f:
    cPickle.dump(opt_theta, f)
    cPickle.dump(zca_white, f)
    cPickle.dump(patch_mean, f)
print('Saved.')

## STEP 2d: Visualize learned features
W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
display_network.display_color_network(W.dot(zca_white).transpose(), 'patches_zca_features.png')