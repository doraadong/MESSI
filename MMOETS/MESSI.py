
import sys
import os
import datetime


import matplotlib
import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.covariance import graphical_lasso

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_regression

from MROTS_soft import MROTS
from fractional_logistic_regression import fractional_logistic_regression
from utils import *

class HME:

    def __init__(self, n_classes_0, n_classes_1, model_name_gates, model_name_experts,
                 group_by_likelihood, idx_conditioned, idx_selected,
                 conditional=False, init_labels = None, soft_classification=True,
                 partial_fit_gate = False, partial_fit_expert = False,
                 stop_by_likelihood = False, tolerance = 3, n_epochs = 20,
                 NEURAL_NET_g = False, NEURAL_NET_e = False):
        '''
        Implementation of hierachical mixture model (Jordan & Jocab,1994) with various
        options of gates or experts.


        Args:
            partial_fit_gate: run for 1 epoch only; with all training samples
            partial_fit_expert: run for 1 epoch only; with all training samples
            soft_classification: 


        '''

        self.n_classes_0 = n_classes_0
        self.n_classes_1 = n_classes_1
        self.GROUP_BY_LIKELIHOOD =  group_by_likelihood
        self.n_epochs = n_epochs
        self.tolerance = tolerance
        self.model_name_gates = model_name_gates
        self.model_name_experts = model_name_experts
        self.init_labels = init_labels
        self.soft = soft_classification
        self.partial_fit_gate = partial_fit_gate
        self.partial_fit_expert = partial_fit_expert
        # self.stop_by_validation = stop_by_validation
        self.MULTITASK = False
        self.FRACTIONAL = False
        self.NEURAL_NET_g  = NEURAL_NET_g
        self.NEURAL_NET_e = NEURAL_NET_e
        self.stop_by_likelihood = stop_by_likelihood 
        self.weights = None # training 
        self.idx_expert = None # testing
        self.probs_expert = None # testing


        # --- Diagnostic only ---
        self.idx_conditioned = idx_conditioned
        self.idx_selected = idx_selected
        self.conditional = conditional

        self._likelihoods_expert_all = []
        self.Y_hat_all = []
        self.labels_1_all = []
        self.gate_probs_all = []
        # ---

        if self.model_name_gates == 'decision_tree':
            if self.partial_fit_gate:
                print(f"Paritial fit not available for decision tree for now.")
            else:
                 self.model_gates = [DecisionTreeClassifier(random_state=0, max_depth=5),
                                   DecisionTreeClassifier(random_state=0, max_depth=5)]

        elif self.model_name_gates == 'logistic':
            if self.partial_fit_gate:
                self.model_gates = [LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial', penalty='l2', max_iter=1,
                                 warm_start=True),
                                   LogisticRegression(random_state=1, solver='lbfgs',
                                 multi_class='multinomial', penalty='l2', max_iter=1,
                                 warm_start=True)]
            else:
                 self.model_gates = [LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial', penalty='l2', max_iter=1e6),
                                   LogisticRegression(random_state=1, solver='lbfgs',
                                 multi_class='multinomial', penalty='l2', max_iter=1e6)]
        elif self.model_name_gates == 'fractional_logistic':
            self.FRACTIONAL = True 
            if self.partial_fit_gate:
                self.model_gates = [fractional_logistic_regression(),
                                    fractional_logistic_regression()]
            else:
                print(f"fractional_logistic now only suppport for partial fit")     

        else:
            print(f"Now only support for decision_tree, logistic or\
            fractional_logistic.")

        self.model_experts = []
        n_experts = self.n_classes_0*self.n_classes_1
        if self.model_name_experts == 'MROTS':
            self.MULTITASK  = True 
            # same initiator for soft or hard for MROTS
            if self.partial_fit_expert:
                for i in range(0,n_experts):
                    lams = [1e-3,1e-3,1e-1,1e-1]
                    self.model_experts.append(MROTS(lams, no_intercept = False, 
                                                    warm_start=True,
                                                    sparse_omega = False, n_iters = 1))
            else:
                for i in range(0,n_experts):
                    lams = [1e-3,1e-3,1e-1,1e-1]
                    self.model_experts.append(MROTS(lams, no_intercept = False,
                                                    warm_start=False,
                                                    sparse_omega = False, n_iters = 20))
        elif self.model_name_experts == 'lasso':
            if self.soft:
                print(f"Weighted learning not available for LASSO for now.")
            else:
                if self.partial_fit_expert:
                    for i in range(0,n_experts):
                        self.model_experts.append(Lasso(alpha=1e-5, precompute=True, max_iter=1,
                                                        warm_start = True, random_state=123,
                                                        selection='random'))
                else:
                    for i in range(0,n_experts):
                        self.model_experts.append(Lasso(alpha=1e-5, precompute=True, max_iter=1000,
                                              random_state=123, selection='random'))

        elif self.model_name_experts == 'linear':
            # same initiator for soft or hard for Linear
            if self.partial_fit_expert:
                print(f"Warining: partial fit for linear regression is same as complete fit when \
                using all training samples to fit")
                for i in range(0,n_experts):
                    self.model_experts.append(LinearRegression())
            else:
                for i in range(0,n_experts):
                    self.model_experts.append(LinearRegression())
        else:
            print(f"Now only support for MROTS,LASSO or Linear Regression.")


    def train_gate(self, model, idx_subset, labels, X_train_clf, current_classes,
               neural_net=False, fractional = False, fractions = None):
        # get subset data for training
        _X = X_train_clf[idx_subset]

        # labels should be the same length as the sub training data
        assert _X.shape[0] == len(labels)

        if len(current_classes) == 1:
            probs = np.ones(X_train_clf.shape[0])
            errors = 0
        else:
            if neural_net:
                EPOCHS = 1000

                # The patience parameter is the amount of epochs to check for improvement
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

                # Use Y_train_sparse for all models for fair comparison
                model.fit(_X, labels, epochs=EPOCHS,
                          validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

                # get prediction for all sample
                probs = model.predict(X_train_clf)

                # get errors for assigned samples
                y_hat_trained = np.where(probs == probs.max(axis=0))[0]  # same order?
                errors = log_loss(labels, y_hat_trained, normalize=True)

            elif fractional and fractions is not None: 
                model.fit(_X, fractions)
                # get prediction for all sample
                probs = model.predict_proba(X_train_clf)

                # get errors for assigned samples
                y_hat_trained = model.predict_proba(_X)
                # errors = cross_entropy_loss(fractions, y_hat_trained)
                errors = log_loss(labels, y_hat_trained, normalize=True)
                
            else:
                model.fit(_X, labels)
                # get prediction for all sample
                probs = model.predict_proba(X_train_clf)

                # get errors for assigned samples
                y_hat_trained = model.predict_proba(_X)
                errors = log_loss(labels, y_hat_trained, normalize=True)

                # make sure the classes for reg same as those for classifier
                assert all(current_classes == model.classes_)

        return probs, errors


    def train_expert(self, model, idx_subset, Y_train, X_train, weight,
                     neural_net=False, MultiTask=False):
        '''

        Args:
            Y_train: 2-dim numpy array
            X_train: 2-dim numpy array
        '''
        # get subset data for training
        _X = X_train[idx_subset, :]
        try:
            _Y = Y_train.iloc[idx_subset, :]
        except:
            _Y = Y_train[idx_subset, :]

        if neural_net:
            EPOCHS = 1000

            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model.fit(_X, _Y, epochs=EPOCHS,
                      validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

            # get prediction for all sample
            Y_hat = model.predict(X_train)

            # get errors for assigned samples
            Y_hat_trained = model.predict(_X)
            errors = (Y_hat_trained - _Y) ** 2

        elif MultiTask:
            model.fit(_X, _Y, sample_weight = weight)

            # get prediction for all samples
            Y_hat = model.predict(X_train)

            # get errors for assigned samples
            Y_hat_trained = model.predict(_X)
            errors = (Y_hat_trained - _Y) ** 2


        elif model == 'zero':

            # get prediction for all samples
            Y_hat = np.zeros_like(Y_train)

             # get errors for assigned samples
            Y_hat_trained = np.zeros_like(_Y)
            errors = (Y_hat_trained - _Y) ** 2

        else:
            # fit for each response y separately
            errors = []
            Y_hat = []

            if _Y.ndim > 1:
                # when response variable has dimension > 1
                for i in range(_Y.shape[1]):
                    # print(f"Currently training and testing gene: {response_list[i]}")
                    try:
                        _y = _Y[:, i]
                    except:
                        _y = _Y.iloc[:, i]

                    model.fit(_X, _y, sample_weight = weight)

                    # get prediction for all samples
                    y_hat = model.predict(X_train)

                    # get errors for assigned samples
                    y_hat_trained = model.predict(_X)
                    error = (y_hat_trained - _y) ** 2

                    Y_hat.append(y_hat)
                    errors.append(error)

            else:
                # when response variable has dimension = 1
                model.fit(_X, _Y, sample_weight = weight)

                # get prediction for all samples
                y_hat = model.predict(X_train)

                # get errors for assigned samples
                y_hat_trained = model.predict(_X)
                error = (y_hat_trained - _Y) ** 2

                Y_hat.append(y_hat)
                errors.append(error)

            errors = np.array(errors)
            Y_hat = np.array(Y_hat).T

        return Y_hat, errors


    def get_expert_probs(self, errors, dist_type=None):
        '''
        Args:
            errors:
                array, number of samples X number of experts

        '''
        if dist_type is None:
            # assume variance = 1
            errors_probs = np.exp(-0.5 * errors)

        return errors_probs


    def get_joint_gate_probs(self, gate_probs_0, gate_probs_1):
        """
        Args:
            gate_probs_0: array, number of samples X number of level 1 output
            gate_probs_1: list of arrays, each containing an array for level 2 gate outputs
                as number of samples X number of level 2 output
        Return:
            gate_probs: array, number of samples X number of experts
        """
        if gate_probs_0.ndim > 1:
            assert gate_probs_0.shape[1] == len(gate_probs_1)
            temp = []
            for i in range(0, gate_probs_0.shape[1]):

                if gate_probs_1[i].ndim == 1:
                    temp.append(gate_probs_0[:, i][:, None] * gate_probs_1[i][:, None])
                else:
                    temp.append(gate_probs_0[:, i][:, None] * gate_probs_1[i])

            return np.concatenate(temp, axis=1)
        else:
            # in this case, gate_probs_0 only has dimension 1 and thus only 1 class for level 0
            assert len(gate_probs_1) == 1
            
            if gate_probs_1[0].ndim == 1:
                return gate_probs_0[:, None] * gate_probs_1[0][:, None]
            else:
                return gate_probs_0[:, None] * gate_probs_1[0]

            

    def get_posterior_expectations(self, gate_probs, expert_probs, cur_iter,
                                   iter_start_gate=0, normalize=False):
        """
        Args:
            gate_probs:  array, number of samples X number of experts
            expert_probs: array, number of samples X number of experts
        """

        _block = np.ones_like(gate_probs)

        if cur_iter < iter_start_gate:
            temp = np.multiply(_block, expert_probs)
        else:
            # temp = ((gate_probs + _block)/2)*expert_probs
            temp = np.multiply(gate_probs, expert_probs)

        if normalize:
            expectations = temp / (temp.sum(axis=1) + 1e-50)[:, None]
            return expectations
        else:
            return temp


    def assign_labels(self, expectations, current_classes):
        '''
        Args:
            expectations: array, number of samples X number of experts

        '''

        best_scores = expectations.max(axis=1)

        # assign level 0 classes to each expert
        _temp = [np.repeat(current_classes[0][i], len(current_classes[i + 1])).flatten()
                 for i in range(len(current_classes[0]))]
        level_0_for_expert = flatten(_temp)

        # assign level 1 classes to each expert
        _temp = [current_classes[i + 1] for i in range(len(current_classes[0]))]
        level_1_for_expert = flatten(_temp)

        # get new labels

        ## level 0 labels
        labels_0 = []

        for i in range(expectations.shape[0]):
            k = np.where(expectations[i, :] == best_scores[i])[0][0]
            labels_0.append(level_0_for_expert[k])

        labels_0 = np.array(labels_0)

        ## level 1 labels
        ### get updated level 0 classes
        current_classes = [i for i in range(0, self.n_classes_0) if i in set(labels_0)]

        labels_1 = []
        for _c in current_classes:
            idx_0 = np.where(labels_0 == _c)[0]
            _labels_1 = []
            for i in idx_0:
                k = np.where(expectations[i, :] == best_scores[i])[0][0]
                _labels_1.append(level_1_for_expert[k])
            labels_1.append(np.array(_labels_1))

        return labels_0, labels_1
    
    def get_objective(self, gate_probs, likelihood_for_all, weights, labels):
        '''
        Calculate the objective negative Q-function. 
        
        '''
        expert_probs = np.array(likelihood_for_all).T
        J = expert_probs.shape[1]
        
        # change to use weightsfor log loss for soft-clustering? 
        _gate = log_loss(labels, gate_probs, normalize=True)
        _temp = np.multiply(weights,np.log(expert_probs+1e-50))
        _experts = - _temp.sum(axis=0)
        
        print(f"Likelihood for gate {_gate}")
        print(f"Likelihood for experts {_experts}")
        return _gate + np.sum(_experts)


    def train(self, X_train, X_train_clf_1, X_train_clf_2, Y_train,
             X_val = None, X_val_clf_1 = None, X_val_clf_2 = None, Y_val = None):
        '''
        Fit gate and experts by training data and learn the labels for each sample.

        TBD:
            1. idx_subset can be infer from weight vector h; no need idx_subset

        '''


        #------ training ------

        N = Y_train.shape[0]
        best_overall_score = 1e9
        best_epoch = 0
        stop_flag = False
        class_collapsed = False

        # initalize lables at level 1
        self.labels_0 = np.random.choice(self.n_classes_0, N, replace=True)

        idx_0 = []
        self.labels_1 = []

        # initalize labels at level 2
        for _c in range(0,self.n_classes_0):
            _idx = np.where(self.labels_0 == _c)[0]
            idx_0.append(_idx)

            if self.init_labels is None:
                _labels = np.random.choice(self.n_classes_1, len(_idx), replace=True)
            else:
                _labels = self.init_labels
            self.labels_1.append(_labels)

        self.labels_all = [self.labels_0] + self.labels_1


        # initialize weight vectors for each expert
        # TBD: now only support weights for 1 single level mixture model
        weights = np.zeros([N, self.n_classes_1])

        for _c in range(self.n_classes_1):
            weights[np.where(self.labels_1[0] == _c)[0], _c] = 1

        assert weights.sum(axis=0).sum() == N
        assert np.all(weights.sum(axis=1) < 2)
        self.weights = weights


        for epoch in range(0, self.n_epochs):
            
#             # --- Diagonostic only ---
            
#             sec_lar =[] 
#             # self.weights not being updated!
#             for i in range(weights.shape[0]):
#                 sec_lar.append(np.partition(weights[i,:].flatten(), -2)[-1])
#             plt.hist(sec_lar)
#             plt.show()
    
#             # ---

            # update classess (asceding order)
            current_classes = []
            for l in range(len(self.labels_all)):
                _labels = self.labels_all[l]
                if l == 0:
                    _cur = [i for i in range(0,self.n_classes_0) if i in set(_labels)]
                else:
                    _cur = [i for i in range(0,self.n_classes_1) if i in set(_labels)]

                current_classes.append(_cur)

            # update idx for clusters
            idx_0 = []
            idx_1 = []
            for i in range(len(current_classes[0])):
                _c = current_classes[0][i]
                _idx_0 = np.where(self.labels_0 == _c)[0]
                idx_0.append(_idx_0)
                # print(f"1st level labels length: {len(_idx_0)}")

                _labels = self.labels_1[i]
                _count = 0
                for j in range(len(current_classes[i+1])):
                    _c = current_classes[i+1][j]
                    _idx_in_subset = np.where(_labels == _c)[0]
                    _idx_in_all = _idx_0[_idx_in_subset]
                    idx_1.append(_idx_in_all)

                    # print(f"2nd level labels length: {len(_idx_in_all)}")
                    _count += len(_idx_in_all)

                # total number assigned to sub-level should equal to upper level
                assert _count == len(_idx_0)
                
            # update weights (no need?)
            if not self.soft:
                weights = np.zeros([N, len(idx_1)])
                
                for _c in range(len(idx_1)):
                    weights[np.where(self.init_labels == _c)[0], _c] = 1

                assert weights.sum(axis=0).sum() == N
                assert np.all(weights.sum(axis=1) < 2)
            

            # maximization of gates
            ## level 1
            model = self.model_gates[0]
            idx_subset = np.arange(0, N)
            labels = self.labels_0
            _class = current_classes[0]

            # print(f"level 1 gate assigned labels length: {len(idx_subset)}")

            probs_0, errors_level_0 = self.train_gate(model, idx_subset, labels,
                                                 X_train_clf_1, _class,
                                                fractional =  self.FRACTIONAL,
                                                      fractions = weights, 
                                            neural_net = self.NEURAL_NET_g)

            ## level 2
            model = self.model_gates[1]

            probs_1 = []
            errors_level_1 = []
            for i in range(len(current_classes[0])):
                idx_subset = idx_0[i]
                labels = self.labels_1[i]
                _class = current_classes[i+1]

                # print(f"level 2 gate assigned labels length: {len(idx_subset)}")
                _probs, _errors = self.train_gate(model, idx_subset, labels,
                                                 X_train_clf_2, _class,
                                                  fractional =  self.FRACTIONAL, 
                                                   fractions = weights, 
                                             neural_net = self.NEURAL_NET_g)
                probs_1.append(_probs)
                errors_level_1.append(_errors)

            # maximization of experts
            errors_experts = []
            errors_for_all = []
            likelihood_for_all = []


            for i in range(len(idx_1)):

                if self.soft:
                    idx_subset = np.arange(0, N)
                    weight = weights[:,i]
                else:
                    idx_subset = idx_1[i]
                    weight = np.ones(len(idx_subset))

                # print(f"expert assigned labels length: {len(idx_subset)}")

                model = self.model_experts[i]
                Y_hat, errors = self.train_expert(model, idx_subset, Y_train,
                                            X_train, weight, MultiTask = self.MULTITASK,
                                            neural_net = self.NEURAL_NET_e)

                errors_experts.append(np.sum(weight[:, None].T @ errors))

                if self.GROUP_BY_LIKELIHOOD:
                    if self.model_name_experts == 'MROTS':
                        _likelihood = model.get_likelihood(Y_hat, Y_train, Y_train.shape[1],
                                                           self.idx_conditioned, conditional = self.conditional)
                        # --- DIAGONOSTIC ONLY ---
                        _likelihoods_expert = model.get_all_likelihood(Y_hat, Y_train, Y_train.shape[1], self.idx_selected)
                        _likelihoods_expert.append(_likelihood)
                        self._likelihoods_expert_all.append(_likelihoods_expert)
                        # ---

                    elif self.model_name_experts == 'lasso' or  self.model_name_experts == 'linear':
                        _sample_var = np.sum((Y_hat - Y_train)**2, axis=0)/np.array(Y_hat.shape[0]-1)
                        _cov = np.diag(_sample_var)
                        _cov_i = np.diag(1/_sample_var)
                        _likelihood = get_multigaussian_pdf(Y_hat, _cov, _cov_i, Y_train.shape[1], Y_train)

                    else:
                        print(f"Now only support for MROTS,Lasso or Linear Regression.")

                    likelihood_for_all.append(_likelihood)


                else:

                    # MSE for each data point, avarage over all response variables
                    try:
                        mse = ((Y_hat - Y_train)**2).mean(axis=1)
                    except IndexError:
                        # a single response variable
                        mse = ((Y_hat - Y_train)**2)

                    errors_for_all.append(mse)
                    
                    
             
            # caculate scores
            if self.stop_by_likelihood:
                # only support for 1 layer gate now 
                _score = self.get_objective(probs_1[0], likelihood_for_all, weights, 
                                           self.labels_1[0])
            else:
                _score = errors_level_0 + sum(errors_level_1) + sum(errors_experts)
                    
            # expectation
            if self.GROUP_BY_LIKELIHOOD:
                expert_probs = np.array(likelihood_for_all).T
                # normalize to 1; sumexp not work for very small likelihood
                expert_probs = expert_probs/(expert_probs.sum(axis=1)+1e-50)[:,None]
                if not np.allclose(expert_probs.sum(axis=1), 1):
                    print (f"Warning: experts probability not sum to 1: {expert_probs.sum(axis=1)}")

            else:
                errors_for_all = np.array(errors_for_all).T
                expert_probs = self.get_expert_probs(errors_for_all, dist_type = None)
                
            gate_probs = self.get_joint_gate_probs(probs_0, probs_1)
            expectations = self.get_posterior_expectations(gate_probs,expert_probs,epoch,
                                                           normalize=True)
             
            # Update weights & labels
            # update weights (no need?)
            if self.soft:
                weights = expectations

            # re-assign labels for all level
            self.labels_0, self.labels_1 = self.assign_labels(expectations,current_classes)
            self.labels_all = [self.labels_0] + self.labels_1

            # --- DIAGONOSTIC ONLY ---
            self.gate_probs_all.append(gate_probs)
            # ---

            # early stopping (either label or score does not further decrease)
            # NOTE: here on training set; should use validation set ideally

#             if self.stop_by_validation:
#                 # not counting in gating error if using validation set
#                 Y_hat_val = self.predict(X_val, X_val_clf_1, X_val_clf_2)
#                 _score = np.sum((Y_hat_val - Y_val) ** 2)

           

            print(f"Best score: {best_overall_score}")
            print(f"Current score: {_score}")
            print(f"level 1 gate error: {errors_level_0}")
            print(f"level 2 gate error: {errors_level_1}")
            print(f"experts error: {errors_experts}")


            if best_overall_score > _score:
                best_overall_score, best_epoch = _score, epoch

            if epoch > best_epoch + self.tolerance:
                stop_flag = True

            if stop_flag:
                break

        print (f"{epoch + 1} iterations in total")
        
        # get final weights (no need?)
        self.weights = weights


    def predict(self, X_test, X_test_clf_1, X_test_clf_2):

        '''
        Use the last saved parameters to make predictions with test data.
        Note now only support for 1 level mixture model.

        '''
        N = X_test.shape[0]

        # update classess (asceding order)
        current_classes = []
        for l in range(len(self.labels_all)):
            _labels = self.labels_all[l]
            if l == 0:
                _cur = [i for i in range(0,self.n_classes_0) if i in set(_labels)]
            else:
                _cur = [i for i in range(0,self.n_classes_1) if i in set(_labels)]

            current_classes.append(_cur)

        # update idx for clusters
        idx_0 = []
        idx_1 = []
        for i in range(len(current_classes[0])):
            _c = current_classes[0][i]
            _idx_0 = np.where(self.labels_0 == _c)[0]
            idx_0.append(_idx_0)

            _labels = self.labels_1[i]
            for j in range(len(current_classes[i+1])):
                _c = current_classes[i+1][j]
                _idx_in_subset = np.where(_labels == _c)[0]
                _idx_in_all = _idx_0[_idx_in_subset]
                idx_1.append(_idx_in_all)
                
        # gates
        ## level 1
        model = self.model_gates[0]
        _class = current_classes[0]

        if len(_class) != 1:
            labels_0_pred = model.predict(X_test_clf_1)
        else:
            labels_0_pred = np.repeat(_class, X_test_clf_1.shape[0])


        ## level 2 gate
        model = self.model_gates[1]
        labels_1_pred_all = []
        
        self.probs_expert = []
        for i in range(len(current_classes[0])):
            _class = current_classes[i+1]
            if len(_class) != 1:
                self.probs_expert.append(model.predict_proba(X_test_clf_2))
                _pred = model.predict(X_test_clf_2)
            else:
                self.probs_expert.append(np.repeat(1,X_test_clf_2.shape[0]))
                _pred = np.repeat(_class,X_test_clf_2.shape[0])
            labels_1_pred_all.append(_pred)
        

        # prepare level 0 classes
        _temp = [np.repeat(current_classes[0][i], len(current_classes[i+1])).flatten()
                       for i in range(len(current_classes[0]))]
        level_0_for_expert = np.array(flatten(_temp))

        # prepare level 1 classes
        _temp = [current_classes[i+1] for i in range(len(current_classes[0]))]
        level_1_for_expert = np.array(flatten(_temp))

        # get expert index
        self.idx_expert = []
        for n in range(X_test.shape[0]):
            _idx = np.where(np.logical_and(level_0_for_expert == labels_0_pred[n],
                                                  level_1_for_expert == labels_1_pred_all[current_classes[0].index(labels_0_pred[n])][n]))[0][0]
            self.idx_expert.append(_idx)

        ## experts
        Y_hat_all = []
        for i in range(len(idx_1)):
            model = self.model_experts[i]
            if self.MULTITASK or self.NEURAL_NET_e:
                Y_hat = model.predict(X_test)
                Y_hat_all.append(Y_hat)

            else:
                print(f"Only support multitask for now!")

        if self.soft:

            ## get the weighted average prediction
            Y_hat_final = np.zeros(Y_hat.shape)
            
            # TBD: generalize to multiple gate 0 classes (Now assume gate 0 1 class)
            cur_probs_expert = self.probs_expert[0]
            
            # TBD: change to einsum
            for i in range(len(idx_1)):
                if cur_probs_expert.ndim ==1:
                    Y_hat_final += np.diag(cur_probs_expert) @ Y_hat_all[i]
                else:
                    Y_hat_final += np.diag(cur_probs_expert[:,i]) @ Y_hat_all[i]

        else:
            Y_hat_final = []
            for n in range(X_test.shape[0]):
                _idx = self.idx_expert[n]
                Y_hat_final.append(Y_hat_all[_idx][n])

        Y_hat_final = np.array(Y_hat_final)

        return Y_hat_final




    def fit_predict(self, X_train, X_test, X_train_clf_1, X_test_clf_1,
                X_train_clf_2, X_test_clf_2, Y_train):
        '''
        Given the labels learned from training, fit the gate & experts with training data
        and make predictions with test data. In this way, only labels need to be saved. 
        
        *TBD: 
            1. adapt for soft prediction


        '''

        #------ prediction ------
        N = Y_train.shape[0]
        weights = self.expectations

        # update classess (asceding order)
        current_classes = []
        for l in range(len(self.labels_all)):
            _labels = self.labels_all[l]
            if l == 0:
                _cur = [i for i in range(0,self.n_classes_0) if i in set(_labels)]
            else:
                _cur = [i for i in range(0,self.n_classes_1) if i in set(_labels)]

            current_classes.append(_cur)

        # update idx for clusters
        idx_0 = []
        idx_1 = []
        for i in range(len(current_classes[0])):
            _c = current_classes[0][i]
            _idx_0 = np.where(self.labels_0 == _c)[0]
            idx_0.append(_idx_0)

            _labels = self.labels_1[i]
            for j in range(len(current_classes[i+1])):
                _c = current_classes[i+1][j]
                _idx_in_subset = np.where(_labels == _c)[0]
                _idx_in_all = _idx_0[_idx_in_subset]
                idx_1.append(_idx_in_all)

        # maximization of gates
        ## level 1
        model = self.model_gates[0]
        idx_subset = np.arange(0, N)
        labels = self.labels_0
        _class = current_classes[0]
        probs_0, errors_level_0 = self.train_gate(model, idx_subset, labels,
                                             X_train_clf_1, _class,
                                                  neural_net=self.NEURAL_NET_g)

        if len(_class) != 1:
            labels_0_pred = model.predict(X_test_clf_1)
        else:
            labels_0_pred = np.repeat(_class, X_test_clf_1.shape[0])

        ## level 2
        model = self.model_gates[1]
        labels_1_pred_all = []
        for i in range(len(current_classes[0])):
            idx_subset = idx_0[i]
            labels = self.labels_1[i]
            _class = current_classes[i+1]
            _probs, _errors  = self.train_gate(model, idx_subset, labels,
                                             X_train_clf_2, _class,
                                               neural_net=self.NEURAL_NET_g)
            if len(_class) != 1:
                _pred = model.predict(X_test_clf_2)
            else:
                _pred = np.repeat(_class,X_test_clf_2.shape[0])

            labels_1_pred_all.append(_pred)

        # prepare level 0 classes
        _temp = [np.repeat(current_classes[0][i], len(current_classes[i+1])).flatten()
                       for i in range(len(current_classes[0]))]
        level_0_for_expert = np.array(flatten(_temp))

        # prepare level 1 classes
        _temp = [current_classes[i+1] for i in range(len(current_classes[0]))]
        level_1_for_expert = np.array(flatten(_temp))

        # get expert index
        self.idx_expert = []
        for n in range(X_test.shape[0]):
            _idx = np.where(np.logical_and(level_0_for_expert == labels_0_pred[n],
                                                  level_1_for_expert == labels_1_pred_all[current_classes[0].index(labels_0_pred[n])][n]))[0][0]
            self.idx_expert.append(_idx)

        # maximization of experts
        Y_hat_all = []
        for i in range(len(idx_1)):

            model = self.model_experts[i]
            if self.soft:
                idx_subset = np.arange(0, N)
                weight = weights[:,i]
            else:
                idx_subset = idx_1[i]
                weight = np.ones(len(idx_subset))

            if self.MULTITASK or self.NEURAL_NET_e:
                _Y_hat_train, _ = self.train_expert(model, idx_subset, Y_train,
                                        X_train, weight, neural_net = self.NEURAL_NET_e,
                                         MultiTask = self.MULTITASK)

                Y_hat = model.predict(X_test)
                Y_hat_all.append(Y_hat)

            else:
                Y_hat = []
                for j in range(0,Y_train.shape[1]):
                    try:
                        y_train = Y_train.iloc[:,j]
                    except:
                        y_train = Y_train[:,j]

                    if y_train.ndim < 2:
                    # when response variable has dimension > 1
                        y_train = y_train[:,None]

                    _, _ = self.train_expert(model, idx_subset, y_train,
                                            X_train, neural_net = self.NEURAL_NET_e,
                                             MultiTask = self.MULTITASK)

                    y_hat = model.predict(X_test)
                    Y_hat.append(y_hat)

                Y_hat_all.append(np.array(Y_hat).T)

        Y_hat_final = []

        for n in range(X_test.shape[0]):
            _idx = self.idx_expert[n]
            Y_hat_final.append(Y_hat_all[_idx][n])

        Y_hat_final = np.array(Y_hat_final)

        return Y_hat_final


    def save_labels(self, current_dir):

        '''
        Save labels for each training sample.

        '''

        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        print(f"Save to: {current_dir}")

        now = datetime.datetime.now()
        now.strftime("%m-%d-%y")

        filename = f"{current_dir}/labels_0_{now.strftime('%m-%d-%y')}"
        np.save(filename, self.labels_0)

        filename = f"{current_dir}/labels_1_{now.strftime('%m-%d-%y')}"
        np.save(filename, self.labels_1)

        filename = f"{current_dir}/labels_all_{now.strftime('%m-%d-%y')}"
        np.save(filename, self.labels_all)


    def save_weights(self, current_dir):


        '''
        Save learned weights.

        '''

        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        print(f"Save to: {current_dir}")

        now = datetime.datetime.now()
        now.strftime("%m-%d-%y")


        for i in range(len(self.model_experts)):
            expert = self.model_experts[i]

            filename = f"{current_dir}/expert_{i}_W_{now.strftime('%m-%d-%y')}"
            np.save(filename, expert.W)

            filename = f"{current_dir}/expert_{i}_b{now.strftime('%m-%d-%y')}"
            np.save(filename, expert.b)

            filename = f"{current_dir}/expert_{i}_omega_{now.strftime('%m-%d-%y')}"
            np.save(filename, expert.omega)

            filename = f"{current_dir}/expert_{i}_sigma_{now.strftime('%m-%d-%y')}"
            np.save(filename, expert.sigma)
