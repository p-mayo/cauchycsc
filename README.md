# Representation Learning via Cauchy Convolutional Sparse Coding
by Perla Mayo, Oktay Karakus, Robin Holmes and Alin M. Achim

The code contained in this repository corresponds to the implementation of the algorithm Cauchy Convolutional Sparse Coding (CCSC) for image reconstruction. The proposed approach  as well as the simulation results can be found  as a preprint [here](https://arxiv.org/abs/2008.03473)

## Abstract

In representation learning, Convolutional Sparse Coding (CSC) enables unsupervised learning of features by jointly optimising both an $\ell_2$-norm fidelity term and a sparsity enforcing penalty. This work investigates using  a regularisation term derived from an assumed Cauchy prior for the coefficients of the feature maps of a CSC generative model. The sparsity penalty term resulting from this prior is solved via its proximal operator, which is then applied iteratively, element-wise, on the coefficients of the feature maps to optimise the CSC cost function. The performance of the proposed Iterative Cauchy Thresholding (ICT) algorithm in reconstructing natural images is compared against algorithms based on  minimising standard penalty functions via soft and hard thresholding as well as against the Iterative Log-Thresholding (ILT) method.


## Implementation
The code has been implemented in Python and requires the following libraries to run:

* Autograd
* Numpy
* Matplotlib
* SciKit
* SciPy
* SparselandTools


## How to run
To execute the program, you need to open a terminal where the code is located in your machine. The main code to test for a given dataset is under the folder csctesting, the file to run is **csc_thresholding.py**, and an example to run it is shown next:

    python -m csctesting.csc_thresholding -dp dataset_path  -bp output_path [-pr prior]

The full list of available arguments and their default values (if none is given), is shown in the table below.

|Parameter|Description|Default value|
|----------|--------|----------|
|-dp --datasetpath | Path of the folder containing the images. It also takes the option "mnist" | * |
| -bp --basepath| Path to save the outputs| * |
| -ns --numsamples | Number of samples to use from the dataset. Value 0 uses all the data available | 0 |
| -nf --numfilters | Number of filters or atoms to learn | 64 |
|-fs --filtersize | Size of the filter. For dimensions greater than 1, this is the size of one side of the filter | 7 |
| --fx --fixed | Whether the program should learn the filters or not. If a string is given, it is assumed no learning is required and it will attempt to load a set of filters from the path specified. For the value of 1 means no learning, value of 2 requires the learning and initialises the filters from random. Any other value initialises the filters from the DCT and requires learning. | 3 |
| -pr --prior | The prior that models the distribution of the coefficients. Possible values: Cauchy, Laplace, Hard, Log | cauchy |
|-lm --lambda| Regularisation parameter | 1|
|-pm --param | Parameter associated to the encoding algorithm (gamma for Cauchy and delta for Log). A value of 0 estimates gamma if the prior is Cauchy and sets delta to 0.001 for Log | 0|
|-lrd --learningrated | Learning rate for the filters | 0.2|
|-lrz --learningratez | Learning rate for the coefficients | 0.015 |
|-mi --maxinner | Maximum number of iterations for the inner steps | 30 |
|-mo --maxouter | Maximum number of iterations for the whole algorithm | 10 |
|-sd --seed| Seed for random value generation. A value of 0 draws a random seed. | 0|

* These parameters are required.
