import numpy as np

#from helpers import math_helpers as mh

def estim_alpha(Y,num_samples):
    """
    We use the estimator in Reconstruction of Ultrasound RF Echoes Modeled as Stable Random Variables
    :param Y: Y is the observations
    :param num_samples: Number of Us used for estimating alpha
    :return: alpha_hat
    """
    m = Y.shape[0]
    alphahat = 0
    count = 0
    while count <= num_samples:
        U = np.random.normal(0,1,[m,1])
        UT = np.transpose(U)
        sig = np.log(abs(np.dot(UT,Y)))
        k1 = np.mean(sig)
        k2 = np.mean((sig[:]-k1)**2)
        if k2 != 0:
            alpha_sq = (2 * np.pi**2 /(12*k2*(1 - np.pi**2/(12*k2))))
            if 0<alpha_sq<4 :
                alpha_sq = alpha_sq ** 0.5
                alphahat += alpha_sq
                count += 1
    alpha = alphahat/count
    return alpha
