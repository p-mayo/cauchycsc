# Representation Learning via Cauchy Convolutional Sparse Coding
by Perla Mayo, Oktay Karakus, Robin Holmes and Alin M. Achim

The code contained in this repository corresponds to the implementation of the algorithm Cauchy Convolutional Sparse Coding (CCSC) for image reconstruction. The proposed approach  as well as the simulation results can be found  as a preprint [here](https://arxiv.org/abs/2008.03473)

## Abstract

In representation learning, Convolutional Sparse Coding (CSC) enables unsupervised learning of features by jointly optimising both an $\ell_2$-norm fidelity term and a sparsity enforcing penalty. This work investigates using  a regularisation term derived from an assumed Cauchy prior for the coefficients of the feature maps of a CSC generative model. The sparsity penalty term resulting from this prior is solved via its proximal operator, which is then applied iteratively, element-wise, on the coefficients of the feature maps to optimise the CSC cost function. The performance of the proposed Iterative Cauchy Thresholding (ICT) algorithm in reconstructing natural images is compared against algorithms based on  minimising standard penalty functions via soft and hard thresholding as well as against the Iterative Log-Thresholding (ILT) method.


## Implementation
The code has been implemented in Python and requires the following libraries to run:

* Pytorch
* Numpy
* Matplotlib
* SciKit
* SciPy


## How to run
To execute the program, you need to open a terminal where the code is located in your machine (**cd cauchycsc_dir**)

    python run_task.py -xml xml_path [-s seed] [-d seed]

The **required** arguments are:

 - xml_path The path of the XML containing all the settings for the execution. Examples of this XML file can be found in [architectures](architectures)

The **optional** arguments are:
- *seed* An initial seed to use. If none is specified, a random seed will be in place
- *dimension* The dimension of the input data to work with. This is not ideal and can be improved, but at this point this is still required

More details on the execution to come...
