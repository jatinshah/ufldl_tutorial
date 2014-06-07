## Stanford Unsupervised Feature Learning and Deep Learning Tutorial

Tutorial Website: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

### Sparse Autoencoder
Sparse Autoencoder vectorized implementation, learning/visualizing features on MNIST data

* [load_MNIST.py](load_MNIST.py): Load MNIST images
* [sample_images.py](sample_images.py): Load sample images for testing sparse auto-encoder
* [gradient.py](gradient.py): Functions to compute & check cost and gradient
* [display_network.py](display_network.py): Display visualized features
* [sparse_autoencoder.py](sparse_autoencoder.py): Sparse autoencoder cost & gradient functions
* [train.py](train.py): Train sparse autoencoder with MNIST data and visualize learnt featured

### Preprocessing: PCA & Whitening
Implement PCA, PCA whitening & ZCA whitening

* [pca_gen.py](pca_gen.py)

### Softmax Regression
Classify MNIST digits via softmax regression (multivariate logistic regression)

* [softmax.py](softmax.py): Softmax regression cost & gradient functions
* [softmax_exercise](softmax_exercise.py): Classify MNIST digits

### Self-Taught Learning and Unsupervised Feature Learning
Classify MNIST digits via self-taught learning paradigm, i.e. learn features via sparse autoencoder using digits 5-9 as unlabelled examples and train softmax regression on digits 0-4 as labelled examples

* [stl_exercise.py](stl_exercise.py): Classify MNIST digits via self-taught learning

### Building Deep Networks for Classification (Stacked Sparse Autoencoder)
Stacked sparse autoencoder for MNIST digit classification

* [stacked_autoencoder.py](stacked_autoencoder.py): Stacked auto encoder cost & gradient functions
* [stacked_ae_exercise.py](stacked_ae_exercise.py): Classify MNIST digits

### Linear Decoders with Auto encoders
Learn features on 8x8 patches of 96x96 STL-10 color images via linear decoder (sparse autoencoder with linear activation function in output layer)

* [linear_decoder_exercise.py](linear_decoder_exercise.py)

### Working with Large Images (Convolutional Neural Networks)
Classify 64x64 STL-10 images using features learnt via linear decoder (previous section) and convolutional neural networks

* [cnn.py](cnn.py): Convolution neural networks. Convolve & Pooling functions
* [cnn_exercise.py](cnn_exercise.py): Classify STL-10 images
