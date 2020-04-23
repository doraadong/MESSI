
import math

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

from utils import *

class MROTS:
    '''
    Implementation of a weighted version of MROTS algorithm based on 
    (Rai, 2012), (Cai, 2014) and Iteratively Reweighted Least Squares(IRLS) algorithm.

    '''
    def __init__(self, lams, no_intercept = False, \
                 sparse_omega = False, sparse_sigma = False, \
                 warm_start = False, n_iters = 20, epsilon = 1e-6):
        self.lam, self.lam1, self.lam2, self.lam3 = lams[0], lams[1],\
        lams[2], lams[3]
        self.no_intercept = no_intercept
        self.sparse_omega = sparse_omega
        self.sparse_sigma = sparse_sigma
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.warm_start = warm_start
        
        # initalize parametes as None
        self.omega, self.sigma, self.omega_i, self.sigma_i ,self.W, self.b  = None, None,\
        None, None, None, None


    def fit(self, X, Y, sample_weight = None):
        '''
        Alternating minimnization.

        Args:
            X, numpy array, N x D
            Y, numpy array, N x K

        '''
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
        
        # get svd for X
        # Not using svd for big sample size

        # U1, s1, V1 = np.linalg.svd(X, full_matrices=True)
        U1, s1, V1 = None, None, None
    
        if self.warm_start and self.omega is not None:
            omega, sigma, omega_i, sigma_i , W, b = \
        self.omega, self.sigma, self.omega_i, self.sigma_i, self.W, self.b
            
        else:
            print('NOT WARM-START!')
            # initialize
            omega = np.identity(K)
            sigma = np.identity(K)
            b = np.mean(Y, axis=0)
            omega_i= np.linalg.inv(omega)
            sigma_i = np.linalg.inv(sigma)

            # get initial W & b
            W = self.update_W(X, Y, U1, s1, V1, omega, omega_i, sigma, sigma_i, 
                              b, D, K, sample_weight = sample_weight)
            if self.no_intercept:
                b = 0
            else:
                b = self.update_b(X, Y, W, sample_weight = sample_weight)

        # get objectives
        pre_obj = self.get_objective(X, Y, omega, omega_i, sigma, \
                                sigma_i, W, b, N, D, sample_weight = sample_weight)
        best_overall_score = pre_obj
        obj.append(pre_obj)
        print(f"Starting objective is {pre_obj}")

        for n in range(0, self.n_iters):
            omega, omega_i = self.update_omega(X, Y, W, b, N, K, sample_weight = sample_weight)
            sigma, sigma_i = self.update_sigma(W, D, K)

            assert is_pos_def(omega)
            assert is_pos_def(sigma)

            W = self.update_W(X, Y, U1, s1, V1, omega, omega_i, sigma, sigma_i, 
                              b, D, K, sample_weight = sample_weight)
            if self.no_intercept:
                b = 0
            else:
                b = self.update_b(X, Y, W, sample_weight = sample_weight)
            cur_obj = self.get_objective(X, Y, omega, omega_i, sigma, \
                                sigma_i, W, b, N, D, sample_weight = sample_weight)
            obj.append(cur_obj)
            print(f"Current objective is {cur_obj}")
            decrease_obj = pre_obj - cur_obj
            pre_obj = cur_obj

            # early stop
#             if  decrease_obj < self.epsilon and n > 1: 
#                 stop_flag = True
                
            if best_overall_score > cur_obj:
                best_overall_score, best_epoch = cur_obj, n

            if n > best_epoch + 1:
                stop_flag = True

            if stop_flag:
                break

        # update parameters
        self.omega, self.sigma, self.omega_i, self.sigma_i ,self.W, self.b = \
        omega, sigma, omega_i, sigma_i, W, b

        return obj

    def get_objective(self, X, Y, omega, omega_i, sigma, \
                                sigma_i, W, b, N, D, sample_weight = None):
        
        H = np.diag(sample_weight)
        
        _dif = Y - X @ W - b 
        _error = np.trace(H @ _dif @ omega_i @ _dif.T)
        _penal = - N*np.linalg.slogdet(omega_i)[1] + self.lam*np.trace(W @ W.T) + \
        self.lam1*np.trace(W @ sigma_i @ W.T) - D*np.linalg.slogdet(sigma_i)[1] + \
        self.lam2*np.sum(abs(omega_i)) + self.lam3*np.sum(abs(sigma_i))
        return _error + _penal

    def update_W(self , X, Y, U1, s1, V1, omega, omega_i, sigma, sigma_i, b, \
                 D, K, svd = False, sample_weight = None):

        '''
        TBD: compare SVD result with Linear system result?

        Two implementations: 1) svd (Cai, 2014), slow for big sample size; assume N>D, N>K
        2) linear system (Rai, 2012), require large memeory for large D & K (DK x DK matrix)

        Args:
            svd:
                assume N > D, N > K
            sample_weight: np array of N x 1 

        '''
        H = np.diag(sample_weight)
        
        if svd:
            P = self.lam*omega + self.lam1*sigma_i @ omega

            # conduct SVD for P
            U2, s2, V2 = np.linalg.svd(P, full_matrices=True)

            S = V1.T @ X.T @ (Y - b) @ U2 # outer product of 1 (N x 1) and b.T
            W_telda = S/((s1**2)[:, None] + (s2**2)[None:, ])
            W = V1 @ W_telda @ U2.T
        else:
            A = np.kron(omega_i, X.T @ H @ X) + np.kron((self.lam1*sigma_i +
                               self.lam*np.identity(K)), np.identity(D))
            b = X.T @ H @ (Y-b) @ omega_i
            W_v = np.linalg.solve(A, b.flatten('F'))
            W = W_v.reshape((-1, K), order='F')
            
        return W

    def update_b(self, X, Y, W, sample_weight = None):
        '''
        Return:
            b: K x 1
        '''
        _sum = (Y-X @ W).T @ sample_weight[:, None]
        _weighted_mean = _sum/np.sum(sample_weight)
        return _weighted_mean.reshape(-1)
        # return np.mean((Y-X @ W), axis=0)

    def update_omega(self, X, Y, W, b, N, K, sample_weight = None):
        _dif = Y - X @ W - b 
        H = np.diag(sample_weight)

        if self.sparse_omega:
            omega, omega_i = graphical_lasso((_dif.T @ _dif)/N, self.lam2)
        else:
            omega = (_dif.T @ H @ _dif + self.lam2*np.identity(K))/np.sum(H)
            omega_i = np.linalg.inv(omega)

        return omega, omega_i

    def update_sigma(self, W, D, K):
        _prod = W.T @ W
        if self.sparse_sigma:
            sigma, sigma_i = graphical_lasso((self.lam1/D)*_prod, self.lam3)
        else:
            sigma = (self.lam1 * _prod + self.lam3*np.identity(K))/D
            sigma_i = np.linalg.inv(sigma)
        return sigma, sigma_i

    def predict(self, X):
        '''
        Linear predictor with learned W and b.

        '''
        return X @ self.W + self.b
    
   
    def get_likelihood(self, Y_hat, Y, K, idx_variable = None, conditional = False):
        '''
        Caculate PDF for all Y; or part of Y conditional on the other part of Y
        with learned parameter.

        Args:
            idx_conditioned: list of integers, index of response variables that
            are conditional on
        '''
        
        # _mean_joint = self.predict(X)
        _mean_joint = Y_hat 
        
        if conditional:
            
            idx_conditioned = [i for i in range(K) if i not in idx_variable]
            
            Y_variable = Y[:,idx_variable]
            Y_conditioned = Y[:,idx_conditioned]
            
            _cov_conditioned = self.omega[idx_conditioned,:][:, idx_conditioned]
            _cov_conditioned_i = np.linalg.inv(_cov_conditioned)
            _cov_vc = self.omega[idx_variable,:][:, idx_conditioned]
            _cov_variable = self.omega[idx_variable,:][:, idx_variable]
            _cov_cv = _cov_vc.T
           
            _mean_variable = _mean_joint[:,idx_variable]
            _mean_conditioned = _mean_joint[:,idx_conditioned]
            
            _cov = _cov_variable - _cov_vc @ _cov_conditioned_i @ _cov_cv
            _mean = _mean_variable + (Y_conditioned - _mean_conditioned) @ (_cov_vc @ _cov_conditioned_i).T
            _cov_i = self.omega_i[:, idx_variable][idx_variable,:]
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
    
    
    def get_all_likelihood(self, Y_hat, Y, K, idx_selected= None):
        '''
        Diagnostic purpose only. Caculate PDF for all Y and for each single Y conditional on the other part of Y
        with learned parameter.

        Args:
            idx_conditioned: list of integers, index of response variables that
            are conditional on
        '''
        
        _mean_joint = Y_hat 
        
        likelihoods = []
        
        # calculate likelihood based on all variables 
        _mean = _mean_joint
        _cov_i = self.omega_i
        _cov = self.omega
        num_variable = K
        Y_variable = Y
            
        # calculate PDF for given mean & cov 
        _det = np.exp(np.linalg.slogdet(_cov)[1])
        _part_a = (Y_variable -_mean) @ _cov_i
        _inner = np.einsum('ij,ij->i', _part_a, (Y_variable -_mean))

        _nominator = np.exp(-.5*_inner)
        _denominator = np.sqrt(np.power(2*math.pi, num_variable)*_det)
        
        _l = _nominator/_denominator
        likelihoods.append(_l)
        
        for i in idx_selected:
            
            idx_variable = [i]
            idx_conditioned = [i for i in range(K) if i not in idx_variable]
            
            Y_variable = Y[:,idx_variable]
            Y_conditioned = Y[:,idx_conditioned]
            
            _cov_conditioned = self.omega[idx_conditioned,:][:, idx_conditioned]
            _cov_conditioned_i = np.linalg.inv(_cov_conditioned)
            _cov_vc = self.omega[idx_variable,:][:, idx_conditioned]
            _cov_variable = self.omega[idx_variable,:][:, idx_variable]
            _cov_cv = _cov_vc.T
           
            _mean_variable = _mean_joint[:,idx_variable]
            _mean_conditioned = _mean_joint[:,idx_conditioned]
            
            _cov = _cov_variable - _cov_vc @ _cov_conditioned_i @ _cov_cv
            _mean = _mean_variable + (Y_conditioned - _mean_conditioned) @ (_cov_vc @ _cov_conditioned_i).T
            _cov_i = self.omega_i[:, idx_variable][idx_variable,:]
            num_variable = len(idx_variable)
            
            # calculate PDF for given mean & cov 
            _det = np.exp(np.linalg.slogdet(_cov)[1])
            _part_a = (Y_variable -_mean) @ _cov_i
            _inner = np.einsum('ij,ij->i', _part_a, (Y_variable -_mean))

            _nominator = np.exp(-.5*_inner)
            _denominator = np.sqrt(np.power(2*math.pi, num_variable)*_det)

            _l = _nominator/_denominator
            
            likelihoods.append(_l)
    
        return likelihoods 
