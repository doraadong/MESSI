import math

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

from messi.utils import *


class mrots:

    def __init__(self, lams, no_intercept=False,
                 sparse_omega=False, warm_start=False, n_iters=20):
        """
        A weighted version of MROTS model based on (Rai, 2012), (Cai, 2014)
        and Iteratively Reweighted Least Squares (IRLS) algorithm.

        Args:
            lams: list of floats, parameters for the regularlizers
            no_intercept: boolean, if include intercept in the model; default included
            sparse_omega: boolean, if assume sparsity for the conditional covariance among responses
            warm_start: boolean, if use the estimates from previous iterations as initialization
            n_iters: integer, number of iterations, default 20
        """
        self.lam, self.lam1, self.lam2, self.lam3 = lams[0], lams[1], \
                                                    lams[2], lams[3]
        print(f"Parameters for regularlizers are: {lams}")
        self.no_intercept = no_intercept
        self.sparse_omega = sparse_omega
        self.n_iters = n_iters
        self.warm_start = warm_start

        # initalize parametes as None
        self.omega, self.sigma, self.omega_i, self.sigma_i, self.W, self.b = None, None, \
                                                                             None, None, None, None

    def fit(self, X, Y, sample_weight=None):
        """
        Train MROTS using alternating minimnization.

        Args:
            sample_weight: numpy array, N x 1, weight of each sample
            X: numpy array, N x D, features
            Y: numpy array, N x K, responses

        Returns:
            obj: list of objectives in every iterations

        """
        # parameter
        N = X.shape[0]
        D = X.shape[1]
        K = Y.shape[1]

        obj = []
        stop_flag = False
        best_epoch = 0

        if sample_weight is None:
            # assign equal weight for all samples 
            sample_weight = np.ones(N)

        if self.warm_start and self.omega is not None:
            omega, sigma, omega_i, sigma_i, W, b = \
                self.omega, self.sigma, self.omega_i, self.sigma_i, self.W, self.b

        else:
            print('NOT WARM-START!')
            # initialize
            omega = np.identity(K)
            sigma = np.identity(K)
            b = np.mean(Y, axis=0)
            omega_i = np.linalg.inv(omega)
            sigma_i = np.linalg.inv(sigma)

            # get initial W & b
            W = self.update_W(X, Y, omega, omega_i, sigma, sigma_i,
                              b, D, K, sample_weight=sample_weight)
            if self.no_intercept:
                b = 0
            else:
                b = self.update_b(X, Y, W, sample_weight=sample_weight)

        # get objectives
        pre_obj = self.get_objective(X, Y, omega, omega_i, sigma,
                                     sigma_i, W, b, N, D, sample_weight=sample_weight)
        best_overall_score = pre_obj
        obj.append(pre_obj)
        print(f"Starting objective is {pre_obj}")

        for n in range(0, self.n_iters):
            omega, omega_i = self.update_omega(X, Y, W, b, N, K, sample_weight=sample_weight)
            sigma, sigma_i = self.update_sigma(W, D, K)

            # omega and sigma supposed to be positive definite
            assert is_pos_def(omega)
            assert is_pos_def(sigma)

            W = self.update_W(X, Y, omega, omega_i, sigma, sigma_i,
                              b, D, K, sample_weight=sample_weight)
            if self.no_intercept:
                b = 0
            else:
                b = self.update_b(X, Y, W, sample_weight=sample_weight)
            cur_obj = self.get_objective(X, Y, omega, omega_i, sigma,
                                         sigma_i, W, b, N, D, sample_weight=sample_weight)
            obj.append(cur_obj)
            print(f"Current objective is {cur_obj}")

            if best_overall_score > cur_obj:
                best_overall_score, best_epoch = cur_obj, n

            if n > best_epoch + 1:
                stop_flag = True

            if stop_flag:
                break

        # update parameters
        self.omega, self.sigma, self.omega_i, self.sigma_i, self.W, self.b = \
            omega, sigma, omega_i, sigma_i, W, b

        return obj

    def get_objective(self, X, Y, omega, omega_i, sigma,
                      sigma_i, W, b, N, D, sample_weight=None):
        """
        Calculate objective given estimates at current iteration.

        Args:
            X: numpy array, N x D, features
            Y: numpy array, N x K, responses
            omega: numpy array, K x K, conditional covariance among responeses
            omega_i: numpy array, K x K, inverse of omega
            sigma: numpy array, K x K, covariance among tasks' coefficients
            sigma_i: numpy array, K x K, inverse of sigma
            W: numpy array, D x K, coefficents of tasks
            b: numpy array, K x 1, intercepts of tasks
            N: integer, sample size
            D: integer, dimension of features
            sample_weight: numpy array, N x 1, weight of each sample

        Returns:
            _obj: float, objective of current iteration
        """

        H = np.diag(sample_weight)

        _dif = Y - X @ W - b
        _error = np.trace(H @ _dif @ omega_i @ _dif.T)
        _penal = - N * np.linalg.slogdet(omega_i)[1] + self.lam * np.trace(W @ W.T) + \
                 self.lam1 * np.trace(W @ sigma_i @ W.T) - D * np.linalg.slogdet(sigma_i)[1] + \
                 self.lam2 * np.sum(abs(omega_i)) + self.lam3 * np.sum(abs(sigma_i))
        _obj = _error + _penal
        return _obj

    def update_W(self, X, Y, omega, omega_i, sigma, sigma_i, b,
                 D, K, sample_weight=None):

        """
        Update coefficents. Two implementations: 1) svd (Cai, 2014), slow for big sample size;
        assume N>D, N>K 2) linear system (Rai, 2012), require large memory for large D & K (DK x DK matrix)
        Here we implemented 2).

        Args:
            X: numpy array, N x D, features
            Y: numpy array, N x K, responses
            omega: numpy array, K x K, conditional covariance among responeses
            omega_i: numpy array, K x K, inverse of omega
            sigma: numpy array, K x K, covariance among tasks' coefficients
            sigma_i: numpy array, K x K, inverse of sigma
            b: numpy array, K x 1, intercepts of tasks
            D: integer, dimension of features
            K: integer, dimension of tasks/responses
            sample_weight: numpy array, N x 1, weight of each sample

        Returns:
            W: numpy array, D x K, coefficents of tasks

        """
        H = np.diag(sample_weight)

        A = np.kron(omega_i, X.T @ H @ X) + np.kron((self.lam1 * sigma_i +
                                                     self.lam * np.identity(K)), np.identity(D))
        b = X.T @ H @ (Y - b) @ omega_i
        W_v = np.linalg.solve(A, b.flatten('F'))
        W = W_v.reshape((-1, K), order='F')

        return W

    def update_b(self, X, Y, W, sample_weight=None):
        """
        Update intercepts.

        Args:
            X: numpy array, N x D, features
            Y: numpy array, N x K, responses
            W: numpy array, D x K, coefficents of tasks
            sample_weight: numpy array, N x 1, weight of each sample
        Return:
            b: K x 1, intercepts
        """
        _sum = (Y - X @ W).T @ sample_weight[:, None]
        _weighted_mean = _sum / np.sum(sample_weight)
        return _weighted_mean.reshape(-1)

    def update_omega(self, X, Y, W, b, N, K, sample_weight=None):

        """
        Update conditional covariance matrix among responeses.

        If assume sparse, then solve by graphical lasso (implemented by scikit_learn).
        Note that this option sometimes encounter warning related to PSD from
        the graphical lasso implementation.

        Args:
            X: numpy array, N x D, features
            Y: numpy array, N x K, responses
            W: numpy array, D x K, coefficents of tasks
            b: numpy array, K x 1, intercepts of tasks
            N: integer, sample size
            K: integer, dimension of tasks/responses
            sample_weight: numpy array, N x 1, weight of each sample

        Returns:
            omega: numpy array, K x K, conditional covariance among responeses
            omega_i: numpy array, K x K, inverse of omega

        """
        _dif = Y - X @ W - b
        H = np.diag(sample_weight)

        if self.sparse_omega:
            omega, omega_i = graphical_lasso((_dif.T @ _dif) / N, self.lam2)
        else:
            omega = (_dif.T @ H @ _dif + self.lam2 * np.identity(K)) / np.sum(H)
            omega_i = np.linalg.inv(omega)

        return omega, omega_i

    def update_sigma(self, W, D, K):
        """
        Update covariance matrix among tasks' coefficients.

        Args:
            W: numpy array, D x K, coefficents of tasks
            D: integer, dimension of features
            K: integer, dimension of tasks/responses

        Returns:
            sigma: numpy array, K x K, covariance among tasks' coefficients
            sigma_i: numpy array, K x K, inverse of sigma

        """
        _prod = W.T @ W

        sigma = (self.lam1 * _prod + self.lam3 * np.identity(K)) / D
        sigma_i = np.linalg.inv(sigma)

        return sigma, sigma_i

    def predict(self, X):
        """
        Linear predictor with learned W and b.

        Args:
            X: numpy array, N x D, features

        Returns:
            _pre: numpy array, N x K, predictions

        """
        _pre = X @ self.W + self.b
        return _pre

    def get_likelihood(self, Y_hat, Y, K, idx_variable=None, conditional=False):
        """
        Calculate likelihood for all responses; or part of responses conditional
        on the other part of responses with learned parameter.

        Args:
            Y_hat: numpy array, N x K, predicted values given by the mean
            Y: numpy array, N x K, responses
            K: integer, dimension of tasks/responses
            idx_variable:  list of integers, index of response variables that are conditional on
            conditional: boolean, if calculate conditional likelihood

        Returns:
            list of likelihoods for every sample
        """

        # take the predicted values as the learned mean
        # _mean_joint = self.predict(X)
        _mean_joint = Y_hat

        if conditional:

            idx_conditioned = [i for i in range(K) if i not in idx_variable]

            Y_variable = Y[:, idx_variable]
            Y_conditioned = Y[:, idx_conditioned]

            _cov_conditioned = self.omega[idx_conditioned, :][:, idx_conditioned]
            _cov_conditioned_i = np.linalg.inv(_cov_conditioned)
            _cov_vc = self.omega[idx_variable, :][:, idx_conditioned]
            _cov_variable = self.omega[idx_variable, :][:, idx_variable]
            _cov_cv = _cov_vc.T

            _mean_variable = _mean_joint[:, idx_variable]
            _mean_conditioned = _mean_joint[:, idx_conditioned]

            _cov = _cov_variable - _cov_vc @ _cov_conditioned_i @ _cov_cv
            _mean = _mean_variable + (Y_conditioned - _mean_conditioned) @ (_cov_vc @ _cov_conditioned_i).T
            _cov_i = self.omega_i[:, idx_variable][idx_variable, :]
            num_variable = len(idx_variable)

            # sanity check (matrix inverse lemma)
            # assert np.allclose(_cov_i, np.linalg.inv(_cov))

        else:
            _mean = _mean_joint
            _cov_i = self.omega_i
            _cov = self.omega
            num_variable = K
            Y_variable = Y

        return get_multigaussian_pdf(_mean, _cov, _cov_i, num_variable, Y_variable)

