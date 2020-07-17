import argparse

import numpy as np
import math

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
    
def get_multigaussian_pdf(_mean, _cov, _cov_i, num_variable, Y_variable):
    """
    calculate multivariate Gaussian PDF for given mean & cov for each sample.

    Parameters
    ----------
        _mean: numpy array
            N x num_variable (N: sample size), mean
        _cov: numpy array
            num_variable x num_variable, covariance matrix
        _cov_i: numpy array
            num_variable x num_variable, inverse of the covariance matrix
        num_variable: integer
            number of variables
        Y_variable: numpy array
            N x num_variable, realized values of the random variables

    Returns
    -------
        likelihoods: list of float
            list of likelihoods for every sample

    Raise
    -----
        NotImplementedError
            Current approach of calculating cannot handle when number of variables larger than 200

    """

    _det = np.exp(np.linalg.slogdet(_cov)[1])
    _part_a = (Y_variable -_mean) @ _cov_i
    _inner = np.einsum('ij,ij->i', _part_a, (Y_variable -_mean))

    _nominator = np.exp(-.5*_inner)
    _denominator = np.sqrt(np.power(2*math.pi, num_variable)*_det)

    if math.isinf(_denominator):
        raise NotImplementedError(f"The current approach suffers from infinity large denominator when number "
                                  f"of variables is large (e.g. > 200)")

    return _nominator/_denominator

flatten = lambda l: [item for sublist in l for item in sublist]


def str2bool(v):
    """
    Helper to pass arguements. Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')