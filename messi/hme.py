import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from mrots import mrots
from utils import *


class hme:

    def __init__(self, n_classes_0, n_classes_1, model_name_gates, model_name_experts, num_responses,
                 group_by_likelihood=True, idx_conditioned=None, conditional=False, init_labels_0=None,
                 init_labels_1=None, soft_weights=True, partial_fit_gate=False, partial_fit_expert=False,
                 tolerance=3, n_epochs=20, random_state=222):
        """
        Implementation of hierachical mixture model (Jordan & Jocab,1994) with various
        options for gates and experts.

        5/3/20:
        a. gates are 'hard', learning on labels (instead of soft as in (Jordan & Jocab,1994))
        b. stopping criterion
            1. based on training set; instead of validation set
            2. based on errors instead of objective Q(,)
        c. in training, if no sample assigned to a class then the class will be removed


        Args:
            n_classes_0: integer, number of classes for the 1st layer; set as 1 if only allow 1 layer
            n_classes_1: integer, number of classes for the 2nd layer; equal to number of experts if only allow 1 layer
            model_name_gates: string, model for the classifiers/gates; any of "decision_tree", "logistic"
            model_name_experts: string, model for the experts; any of "mrots", "lasso" or "linear".
            num_responses: integer, number of response variables
            group_by_likelihood: boolean, if samples assigned to experts by likelihood; default true
            idx_conditioned: list of integers, index of response variables that are conditional for likelihood; default none
            conditional: boolean, if likelihood is conditional or not; default false
            init_labels_0: list of cluster labels (0,1,..) for level 1, N x 1, initialized grouping of samples; default none
            init_labels_1: list of list of cluster labels (0,1,..) for level 2, initialized grouping of samples; default none
            soft_weights: boolean, if apply soft grouping of samples; default true
            partial_fit_gate: boolean, run for 1 epoch only, using all training samples; default false
            partial_fit_expert: boolean, run for 1 epoch only, using all training samples; default false
            tolerance: integer, number of epochs to run after reaching the best epoch before stopping; default 3
            n_epochs: integer, total number of epochs; default 20
            random_state: integer, random state applying to numpy.random.choice or experts/classifiers

        Raises:
            NotImplementedError: some experts or gates models are not supported
        """

        self.n_classes_0 = n_classes_0
        self.n_classes_1 = n_classes_1
        self.model_name_gates = model_name_gates
        self.model_name_experts = model_name_experts
        self.num_responses = num_responses
        self.GROUP_BY_LIKELIHOOD = group_by_likelihood
        self.idx_conditioned = idx_conditioned
        self.conditional = conditional
        self.init_labels_0 = init_labels_0
        self.init_labels_1 = init_labels_1
        self.soft = soft_weights
        self.partial_fit_gate = partial_fit_gate
        self.partial_fit_expert = partial_fit_expert
        self.n_epochs = n_epochs
        self.tolerance = tolerance
        self.random_state = random_state
        self.MULTITASK = False  # if expert is multi-task; default false
        self.weights = None  # expert weights; used in training
        self.labels_0 = None  # sample labels; used in training
        self.labels_1 = None  # sample labels; used in training
        self.labels_all = None  # sample labels; used in training
        self.idx_expert = None  # expert labels; used in prediction
        self.probs_expert = None  # expert weights; used in prediction

        # initialize gates
        # -1 for the level 1 gate; level 2 gates are labeled by level 1 classes
        _keys = [-1] + list(range(self.n_classes_0))

        if self.model_name_gates == 'decision_tree':
            if self.partial_fit_gate:
                raise NotImplementedError(f"Paritial fit not available for decision tree for now.")
            else:
                self.model_gates = {key: DecisionTreeClassifier(random_state=self.random_state, max_depth=5)
                                    for key in _keys}

        elif self.model_name_gates == 'logistic':
            if self.partial_fit_gate:
                _max_iter = 1
            else:
                _max_iter = 1e6

                self.model_gates = {key: LogisticRegression(random_state=self.random_state, solver='lbfgs',
                                                            multi_class='multinomial', penalty='l2', max_iter=_max_iter)
                                    for key in _keys}
        else:
            raise NotImplementedError(f"Now only support decision_tree or logistic")

        # initialize experts
        self.model_experts = defaultdict(dict)

        # initialize the number of experts, equals to the product of the number of classes in layer 1 & layer 2
        self.n_experts = self.n_classes_0 * self.n_classes_1

        if self.model_name_experts == 'mrots':
            lams = [1e-3, 1e-3, 1e-1, 1e-1]  # hyper-parameters for MROTS
            self.MULTITASK = True
            # same initiator for soft or hard for MROTS
            if self.partial_fit_expert:
                _warm_star = True
                _n_iters = 1
            else:
                _warm_star = False
                _n_iters = 20

            for i in range(self.n_classes_0):
                for j in range(self.n_classes_1):
                    self.model_experts[i][j] = mrots(lams, no_intercept=False, warm_start=_warm_star,
                                                     sparse_omega=False, n_iters=_n_iters)

        elif self.model_name_experts == 'lasso':
            if self.soft:
                raise NotImplementedError(f"Weighted learning not available for lasso for now.")
            else:
                if self.partial_fit_expert:
                    _max_iter = 1
                    _warm_start = True
                else:
                    _max_iter = 1000
                    _warm_start = False

                for i in range(self.n_classes_0):
                    for j in range(self.n_classes_1):
                        _models = [Lasso(alpha=1e-5, precompute=True, max_iter=_max_iter,
                                         warm_start=_warm_start, random_state=self.random_state,
                                         selection='random') for i in range(self.num_responses)]
                        self.model_experts[i][j] = _models

        elif self.model_name_experts == 'linear':
            # same initiator for soft or hard for Linear
            if self.partial_fit_expert:
                warnings.warn(f"Warining: partial fit for linear regression is same as complete fit when \
                using all training samples to fit")

            for i in range(self.n_classes_0):
                for j in range(self.n_classes_1):
                    _models = [LinearRegression() for i in range(self.num_responses)]
                    self.model_experts[i][j] = _models

        else:
            raise NotImplementedError(f"Now only support for mrots, lasso or linear.")

    @staticmethod
    def train_gate(model, idx_subset, labels, X_train_clf, current_classes):
        """

        Train a single gate/classifier.

        Args:
            model: object from a model class
            idx_subset: list of integers, indexes of samples that are assigned to this gate (and used for training)
            labels: list of integers, labels of this subset of samples
            X_train_clf: numpy array, sample size x feature size, features for the gate
            current_classes: list of integers, unique labels of non-empty classes

        Returns:
            probs: numpy array, (total) sample size x number of classes, predicted probability of samples assigned \
            to different classes
            errors: float, cross-entropy error on this subset of training samples

        """
        # get subset data for training
        _X = X_train_clf[idx_subset]

        # number of samples in labels should be the same as the sub training data
        assert _X.shape[0] == len(labels)

        if len(current_classes) == 1:
            probs = np.ones(X_train_clf.shape[0])
            errors = 0
        else:
            model.fit(_X, labels)
            # get predictions for all samples
            probs = model.predict_proba(X_train_clf)

            # get errors for the subset of samples
            y_hat_trained = model.predict_proba(_X)
            errors = log_loss(labels, y_hat_trained, normalize=True)

            # make sure the class labels used by the classifier same as those in current_classes
            assert all(current_classes == model.classes_)

        return probs, errors

    @staticmethod
    def train_expert(model, idx_subset, Y_train, X_train, weight, MultiTask=False):
        """

        Train a single expert.

        Args:
            model: object or a list of objects (if MultiTask = false) from a model class
            idx_subset: list of integers, indexes of samples that are assigned to this expert (and used for training)
            Y_train: numpy array, sample size x number of responses, response variables
            X_train: numpy array, sample size x feature size, features for experts
            weight: numpy array, sample size x 1, sample weights for this expert
            MultiTask: boolean, if the expert is multi-task

        Returns:
            Y_hat: numpy array, (total) sample size x number of responses, predictions from this expert
            errors: float, (unweighted) mean square error on the assigned training samples

        """
        # get subset data for training
        _X = X_train[idx_subset, :]
        _Y = Y_train[idx_subset, :]

        if MultiTask:
            model.fit(_X, _Y, sample_weight=weight)

            # get prediction for all samples
            Y_hat = model.predict(X_train)

            # get errors for assigned samples
            Y_hat_trained = model.predict(_X)
            errors = (Y_hat_trained - _Y) ** 2

        else:
            errors = []
            Y_hat = []

            K = 1
            # when response variable has dimension > 1
            if _Y.ndim > 1:
                K = _Y.shape[1]

            # number of sub-models equal to number of responses
            assert K == len(model)

            # fit for each response y separately
            for i in range(K):
                # print(f"Currently training and testing gene: {response_list[i]}")
                if _Y.ndim > 1:
                    _y = _Y[:, i]
                else:
                    _y = _Y

                sub_model = model[i]
                sub_model.fit(_X, _y, sample_weight=weight)

                # get prediction for all samples
                y_hat = sub_model.predict(X_train)

                # get errors for assigned samples
                y_hat_trained = sub_model.predict(_X)
                error = (y_hat_trained - _y) ** 2

                Y_hat.append(y_hat)
                errors.append(error)

            errors = np.array(errors)
            Y_hat = np.array(Y_hat).T

        return Y_hat, errors

    @staticmethod
    def get_expert_probs(errors, dist_type=None):
        """
        Transform mean square error into a 'probability' form. (Not real probability)

        Args:
            errors: numpy array, sample size x number of experts, mean square errors
            dist_type: string, type of distribution; default none

        Returns:
            errors_probs: umpy array, sample size x number of experts, transformed errors

        """
        if dist_type is None:
            # assume variance = 1
            errors_probs = np.exp(-0.5 * errors)

        return errors_probs

    @staticmethod
    def get_joint_gate_probs(gate_probs_0, gate_probs_1):
        """

        Calculate the joint probability of classes across 2 levels (equal to the probability of experts).
        That is: p(level 1 = i, level 2 = j| x) = p(expert = z | x)

        Args:
            gate_probs_0: numpy array, sample size x number of level 1 classes, level 1 gate outputs
            gate_probs_1: list of numpy arrays, each containing an array of sample size x number of \
            level 2 classes corresponding to the outputs from a level 2 gate

        Return:
            gate_probs: numpy array, sample size x number of experts
        """

        # if multiple classes exist for level 1
        if gate_probs_0.ndim > 1:
            # number of level 1 classes equal to the number of level 2 gates
            assert gate_probs_0.shape[1] == len(gate_probs_1)
            temp = []
            for i in range(0, gate_probs_0.shape[1]):
                # for the i_th level 1 class, the joint probability equals to
                # p(level 1=i | x) * p(level 2 | level 1=i, x)
                if gate_probs_1[i].ndim == 1:
                    temp.append(gate_probs_0[:, i][:, None] * gate_probs_1[i][:, None])
                else:
                    temp.append(gate_probs_0[:, i][:, None] * gate_probs_1[i])

            return np.concatenate(temp, axis=1)
        else:
            # in this case, gate_probs_0 only has dimension 1 and thus only 1 class for level 1
            # and 1 gate for level 2
            assert len(gate_probs_1) == 1

            if gate_probs_1[0].ndim == 1:
                return gate_probs_0[:, None] * gate_probs_1[0][:, None]
            else:
                return gate_probs_0[:, None] * gate_probs_1[0]

    @staticmethod
    def get_posterior_expectations(gate_probs, expert_probs, cur_iter,
                                   iter_start_gate=0, normalize=True):
        """

        Calculate the posterior probability/weight of experts as the product of likelihood and expert probability.
        That is: p(expert = z | x, y) = p(y | expert = z, x) * p(expert = z | x) / p(y | x)

        Args:
            gate_probs: numpy array, sample size x number of experts, joint probability of classes given features
            expert_probs: numpy array, sample size x number of experts, sample likelihood of each expert
            cur_iter: integer, current epoch
            iter_start_gate: integer, before which use only expert likelihood to approximate the weight; default 0
            normalize: boolean, if normalize by p(y | x) or not, default true

        Returns:
            posterior weights of experts (normalized or not), numpy array, sample size x number of experts

        """

        _block = np.ones_like(gate_probs)

        # early learning for gates is not stable sometimes; thus use only likelihood
        if cur_iter < iter_start_gate:
            temp = np.multiply(_block, expert_probs)
        else:
            # temp = ((gate_probs + _block)/2)*expert_probs
            temp = np.multiply(gate_probs, expert_probs)

        if normalize:
            # TBD: might be a better way than taking just 1e-50
            # have tried sum-exp but it did not work for very small likelihood
            expectations = temp / (temp.sum(axis=1) + 1e-50)[:, None]
            return expectations
        else:
            return temp

    @staticmethod
    def map_classes_to_expert(current_classes):
        """

        Map classes at different levels with the index of experts (with index 0,1,2,...)
        based on a set of available (non-empty) classes at different levels.

        Args:
            current_classes: list of list of integers, each sub-list corresponnds to a gate, containing unique labels \
            of non-empty classes

        Returns:
            level_0_for_expert: list of integers, the class at level 1 for an expert
            level_1_for_expert: list of integers, the class at level 2 for an expert

        """
        # map level 1 classes to each expert
        _temp = [np.repeat(current_classes[0][i], len(current_classes[i + 1])).flatten()
                 for i in range(len(current_classes[0]))]
        level_0_for_expert = flatten(_temp)

        # map level 2 classes to each expert
        _temp = [current_classes[i + 1] for i in range(len(current_classes[0]))]
        level_1_for_expert = flatten(_temp)

        # both equal to the current number of experts
        assert len(level_0_for_expert) == len(level_1_for_expert)

        return level_0_for_expert, level_1_for_expert

    def assign_labels(self, expectations, current_classes):
        """

        Re-assign labels for all levels of classes to samples based on estimated weights of experts.
        Note this is different from the original algorithm in (Jordan & Jocab,1994) where they use
        directly the weights to fit gates. Here we assume that the expert with highest weight (say expert j)
        takes all probability mass (i.e. p(expert = j | y, x) = 1 and p(expert = others | y, x) = 0).
        Then, for example, if (expert = 2) = (level 1 = 0, level 2 = 2); the class at level 2
        for this sample is 2 and the class at level 1 for this sample is 0.  In this way,
        we can fit gates with labels (thus hard classification).

        Args:
            expectations: numpy array, sample size x number of experts, weights of experts
            current_classes: list of list of integers, each sub-list corresponnds to a gate, containing unique labels \
            of non-empty classes

        Returns:
            labels_0: list of labels (0,1,..), N x 1, new labels based on learned expectations for level 1
            labels_1: list of list of labels (0,1,..), each sub-list for each level 2 gate, \
            new labels based on learned expectations for level 2

        """
        # get the largest weight for each sample
        best_scores = expectations.max(axis=1)

        # map classes to expert index
        level_0_for_expert, level_1_for_expert = self.map_classes_to_expert(current_classes)

        # get new level 1 labels
        labels_0 = []

        for i in range(expectations.shape[0]):
            k = np.where(expectations[i, :] == best_scores[i])[0][0]
            labels_0.append(level_0_for_expert[k])

        labels_0 = np.array(labels_0)

        # get new level 2 labels
        # update the set of non-empty level 1 classes based on new level 1 assignments
        current_classes_0 = [i for i in range(0, self.n_classes_0) if i in set(labels_0)]

        labels_1 = []
        for _c in current_classes_0:
            idx_0 = np.where(labels_0 == _c)[0]
            _labels_1 = []
            for i in idx_0:
                k = np.where(expectations[i, :] == best_scores[i])[0][0]
                _labels_1.append(level_1_for_expert[k])
            labels_1.append(np.array(_labels_1))

        return labels_0, labels_1

    def get_current_classes(self):
        """

        Update non-empty classes (in ascending order) for each layer, based on newly assigned labels inferred from the
        posterior weights of experts.

        Returns:
            current_classes: list of list of integers, each sub-list corresponnds to a gate, containing unique labels \
            of non-empty classes
            n_experts: integer, number of experts based on available classes

        """
        current_classes = []
        for l in range(len(self.labels_all)):
            _labels = self.labels_all[l]
            if l == 0:
                _cur = [i for i in range(0, self.n_classes_0) if i in set(_labels)]
            else:
                _cur = [i for i in range(0, self.n_classes_1) if i in set(_labels)]

            current_classes.append(_cur)

        # number of level 1 classes equal to the number of gates at level 2
        assert len(current_classes[0]) == len(current_classes[1:])

        # get the number of terminal classes (available experts)
        n_experts = len(flatten(current_classes[1:]))

        return current_classes, n_experts

    def map_sample_to_class(self, current_classes):
        """

        Map sample index to the classes they were assigned to at different levels.

        Args:
            current_classes: list of list of integers, each sub-list corresponnds to a gate, containing unique labels \
            of non-empty classes

        Returns:
            idx_0: list of list of integers, each sub-list corresponds to the samples assigned to a class at level 1
            idx_1: list of list of integers, each sub-list corresponds to the samples assigned to a class at level 2

        """

        # update idx of samples belonging to different classes
        idx_0 = []
        idx_1 = []
        for i in range(len(current_classes[0])):
            _c = current_classes[0][i]
            _idx_0 = np.where(self.labels_0 == _c)[0]
            idx_0.append(_idx_0)
            # print(f"1st level labels length: {len(_idx_0)}")

            _labels = self.labels_1[i]
            _count = 0
            for j in range(len(current_classes[i + 1])):
                _c = current_classes[i + 1][j]
                _idx_in_subset = np.where(_labels == _c)[0]
                _idx_in_all = _idx_0[_idx_in_subset]
                idx_1.append(_idx_in_all)

                # print(f"2nd level labels length: {len(_idx_in_all)}")
                _count += len(_idx_in_all)

            # total number assigned to sub-level should equal to upper level
            assert _count == len(_idx_0)

        return idx_0, idx_1

    def train(self, X_train, X_train_clf_1, X_train_clf_2, Y_train):
        """
        Train gates and experts based on EM.

        Args:
            X_train: numpy array, sample size x feature size, features for experts
            X_train_clf_1: numpy array, sample size x feature size, features for the 1st layer gate
            X_train_clf_2: numpy array, sample size x feature size, features for the 2nd layer gates
            Y_train: numpy array, sample size x number of responses, response variables

        Raises:
            NotImplementedError: some experts or gates models are not supported

        """

        # ------ setting parameters ------
        N = Y_train.shape[0]
        best_overall_score = 1e9
        best_epoch = 0
        stop_flag = False

        # input has same number response variables as initialized
        assert Y_train.shape[1] == self.num_responses

        # initialize labels at level 1
        if self.init_labels_0 is None:
            # if labels not given; initialize randomly
            rng = np.random.RandomState(self.random_state)
            self.labels_0 = rng.choice(self.n_classes_0, N, replace=True)
        else:
            self.labels_0 = self.init_labels_0

        idx_0 = []
        self.labels_1 = []

        # initialize labels at level 2
        for _c in range(0, self.n_classes_0):
            _idx = np.where(self.labels_0 == _c)[0]
            idx_0.append(_idx)

            if self.init_labels_1 is None:
                # if labels not given; initialize randomly
                rng = np.random.RandomState(self.random_state)
                _labels = rng.choice(self.n_classes_1, len(_idx), replace=True)
            else:
                _labels = self.init_labels_1[_c]
            self.labels_1.append(_labels)

        self.labels_all = [self.labels_0] + self.labels_1

        # update non-empty classes (ascending order) for each layer
        current_classes, self.n_experts = self.get_current_classes()

        # update idx of samples belonging to different classes
        idx_0, idx_1 = self.map_sample_to_class(current_classes)

        # initialize weight vectors for each expert based on the initialized labels (hard grouping)
        if self.soft:
            self.weights = np.zeros([N, self.n_experts])

            for _c in range(self.n_experts):
                self.weights[idx_1[_c], _c] = 1

            assert self.weights.sum(axis=0).sum() == N  # if each sample has been assigned to an expert
            assert np.all(self.weights.sum(axis=1) < 2)  # if each sample has been assigned to an single expert

        for epoch in range(0, self.n_epochs):

            # maximization of gates/classifiers
            # level 1 (1 single gate)
            model = self.model_gates[-1]
            idx_subset = np.arange(0, N)
            labels = self.labels_0
            _class = current_classes[0]

            # print(f"level 1 gate assigned labels length: {len(idx_subset)}")
            probs_0, errors_level_0 = self.train_gate(model, idx_subset, labels,
                                                      X_train_clf_1, _class)

            # level 2
            probs_1 = []
            errors_level_1 = []

            # train each level 2 gate one-by-one
            for i in range(len(current_classes[0])):
                _c = current_classes[0][i]  # get the class at level 1
                model = self.model_gates[_c]
                idx_subset = idx_0[i]
                labels = self.labels_1[i]
                _class = current_classes[i + 1]

                # print(f"level 2 gate assigned labels length: {len(idx_subset)}")
                _probs, _errors = self.train_gate(model, idx_subset, labels,
                                                  X_train_clf_2, _class)
                probs_1.append(_probs)
                errors_level_1.append(_errors)

            # maximization of experts
            errors_experts = []
            errors_for_all = []
            likelihood_for_all = []

            # train each expert one-by-one
            i = 0  # index for expert
            for j in current_classes[0]:  # level 1
                c0 = current_classes[0][j]
                for c1 in current_classes[j + 1]:  # level 2
                    if self.soft:
                        idx_subset = np.arange(0, N)  # use full-set of training samples
                        weight = self.weights[:, i]
                    else:
                        idx_subset = idx_1[i]  # use subset of training samples
                        weight = np.ones(len(idx_subset))  # set weight all being 1

                    model = self.model_experts[c0][c1]
                    Y_hat, errors = self.train_expert(model, idx_subset, Y_train,
                                                      X_train, weight, MultiTask=self.MULTITASK)

                    # weighting the errors from this expert
                    errors_experts.append(np.sum(weight[:, None].T @ errors))

                    if self.GROUP_BY_LIKELIHOOD:
                        # use assumed distribution (here multivariate Gaussian) for posterior probability
                        if self.model_name_experts == 'mrots':
                            # calculate likelihood with multivaraite Gaussian assumming dependence among responses
                            _likelihood = model.get_likelihood(Y_hat, Y_train, self.num_responses,
                                                               self.idx_conditioned, conditional=self.conditional)

                        elif self.model_name_experts == 'lasso' or self.model_name_experts == 'linear':
                            # calculate likelihood with multivaraite Gaussian assumming independence among responses
                            _sample_var = np.sum((Y_hat - Y_train) ** 2, axis=0) / np.array(Y_hat.shape[0] - 1)
                            _cov = np.diag(_sample_var)
                            _cov_i = np.diag(1 / _sample_var)
                            _likelihood = get_multigaussian_pdf(Y_hat, _cov, _cov_i, Y_train.shape[1], Y_train)

                        else:
                            raise NotImplementedError(f"Now only support for mrots, lasso or linear.")

                        likelihood_for_all.append(_likelihood)

                    else:
                        # grouped based on average MSE (MSE for each data point, average over all response variables)
                        try:
                            mse = ((Y_hat - Y_train) ** 2).mean(axis=1)
                        except IndexError:
                            # a single response variable
                            mse = ((Y_hat - Y_train) ** 2)

                        errors_for_all.append(mse)

                    i += 1

            # calculate scores based on errors
            _score = errors_level_0 + sum(errors_level_1) + sum(errors_experts)

            if self.GROUP_BY_LIKELIHOOD:
                expert_probs = np.array(likelihood_for_all).T
            else:
                # transform MSE into a 'probability' like form
                errors_for_all = np.array(errors_for_all).T
                expert_probs = self.get_expert_probs(errors_for_all, dist_type=None)

            # calculate posterior probability/weights of experts
            gate_probs = self.get_joint_gate_probs(probs_0, probs_1)  # get joint probability of classes given features
            expectations = self.get_posterior_expectations(gate_probs, expert_probs, epoch,
                                                           normalize=True)
            if not np.allclose(expectations.sum(axis=1), 1):
                warnings.warn(f"experts weights not sum to 1 for some samples!")

            # update the posterior probability
            if self.soft:
                self.weights = expectations

            # re-assign labels based on the new posterior probability
            self.labels_0, self.labels_1 = self.assign_labels(expectations, current_classes)
            self.labels_all = [self.labels_0] + self.labels_1

            # update non-empty classes (ascending order) for each layer
            current_classes, self.n_experts = self.get_current_classes()

            # update idx of samples belonging to different classes
            idx_0, idx_1 = self.map_sample_to_class(current_classes)

            # early stopping (either label or score does not further change)
            # NOTE: here stopping is based on training set;

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

            print(f"{epoch + 1} epochs in total")

    def predict(self, X_test, X_test_clf_1, X_test_clf_2):

        """
        Make predictions on test data.

        Args:
            X_test: numpy array, sample size x feature size, features for experts
            X_test_clf_1: numpy array, sample size x feature size, features for the 1st layer gate
            X_test_clf_2: numpy array, sample size x feature size, features for the 2nd layer gates

        Returns:
            Y_hat_final: numpy array, sample size x number of responses, predictions

        Raises:
            NotImplementedError: some experts or gates models are not supported

        """
        # ------ setting parameters ------
        N = X_test.shape[0]

        # update non-empty classes (ascending order) for each layer
        current_classes, self.n_experts = self.get_current_classes()

        # map classes to expert index
        level_0_for_expert, level_1_for_expert = self.map_classes_to_expert(current_classes)
        level_0_for_expert, level_1_for_expert = np.array(level_0_for_expert), np.array(level_1_for_expert)

        # get predictions from level 1 gates
        model = self.model_gates[-1]
        _class = current_classes[0]

        if len(_class) != 1:
            labels_0_pred = model.predict(X_test_clf_1)
            probs_0 = model.predict_proba(X_test_clf_1)
        else:
            labels_0_pred = np.repeat(_class[0], X_test_clf_1.shape[0])
            probs_0 = np.repeat(1, X_test_clf_1.shape[0])

        # get predictions from level 2 gates
        labels_1_pred_all = []
        probs_1 = []

        for i in range(len(current_classes[0])):
            _c = current_classes[0][i]  # get the class at level 1
            model = self.model_gates[_c]
            _class = current_classes[i + 1]

            if len(_class) != 1:
                _prob = model.predict_proba(X_test_clf_2)
                _pred = model.predict(X_test_clf_2)
            else:
                _prob = np.repeat(1, X_test_clf_2.shape[0])
                _pred = np.repeat(_class, X_test_clf_2.shape[0])

            labels_1_pred_all.append(_pred)
            probs_1.append(_prob)

        # get joint classes probability (expert probability)
        if self.soft:
            self.probs_expert = self.get_joint_gate_probs(probs_0, probs_1)

        # get expert index for each sample
        self.idx_expert = []
        for n in range(X_test.shape[0]):
            _idx = np.where(np.logical_and(level_0_for_expert == labels_0_pred[n],
                                           level_1_for_expert ==
                                           labels_1_pred_all[current_classes[0].index(labels_0_pred[n])][n]))[0][0]
            self.idx_expert.append(_idx)

        # get predictions from experts
        Y_hat_all = []
        i = 0  # index for expert
        for j in current_classes[0]:  # level 1
            c0 = current_classes[0][j]
            for c1 in current_classes[j + 1]:  # level 2
                model = self.model_experts[c0][c1]
                if self.MULTITASK:
                    Y_hat = model.predict(X_test)
                elif self.model_name_experts == 'lasso' or self.model_name_experts == 'linear':
                    # number of sub-models equal to number of responses
                    assert self.num_responses == len(model)

                    Y_hat = []

                    # get for each response y separately
                    for k in range(self.num_responses):
                        sub_model = model[k]
                        y_hat = sub_model.predict(X_test)
                        Y_hat.append(y_hat)

                    Y_hat = np.array(Y_hat).T
                else:
                    raise NotImplementedError(f"Now only support for mrots, lasso or linear.")

                Y_hat_all.append(Y_hat)
                i += 1

        # get weighted average based on the predictions from experts and gates
        if self.soft:
            # get the weighted average prediction
            Y_hat_final = np.zeros(Y_hat.shape)

            # apply weight for each experts' output and get the summation
            # TBD: change to einsum
            for i in range(self.n_experts):
                if self.probs_expert.ndim == 1:
                    Y_hat_final += np.diag(self.probs_expert) @ Y_hat_all[i]
                else:
                    Y_hat_final += np.diag(self.probs_expert[:, i]) @ Y_hat_all[i]

        else:
            # choose the best from all experts' output
            Y_hat_final = []
            for n in range(X_test.shape[0]):
                _idx = self.idx_expert[n]
                Y_hat_final.append(Y_hat_all[_idx][n])

        Y_hat_final = np.array(Y_hat_final)

        return Y_hat_final



