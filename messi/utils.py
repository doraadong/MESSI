
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

    """

    _det = np.exp(np.linalg.slogdet(_cov)[1])
    _part_a = (Y_variable -_mean) @ _cov_i
    _inner = np.einsum('ij,ij->i', _part_a, (Y_variable -_mean))

    _nominator = np.exp(-.5*_inner)
    _denominator = np.sqrt(np.power(2*math.pi, num_variable)*_det)

    return _nominator/_denominator

flatten = lambda l: [item for sublist in l for item in sublist]


