import math
from collections import defaultdict
import copy
import statistics
import itertools

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy import stats

from messi.data_processing import *


class gridSearch:
    """

    For discrete parameters, search for the best by cross-validation specifically designed for single cell spatial
    data where multiple independent animal/samples exist and spatial relationship among cells within a single sample
    need to preserved. A single model object will be used with updated parameter values each time.


    Parameters
    ----------
        params_dict: dictionary
            parameter's name as key (e.g. "n_classes_1", "alpha") and the list of possible values in the \
            ascending order as the value
        model: object
            an object from a messi.hme class or a model class implementing method fit(), predict() (e.g. sklearn models)
        ratio: float
            the ratio of number of validation datasets / all datasets in a single sample
        n_sets: integer
            number of validations sets; also the number of cross-validations to run
        params_list: list
            list of dictionary, each element specifies a combination of parameter values with key as (e.g. \
            "n_classes_1", "alpha") and a specific value; default None
        random_state: integer
            random state applying to numpy.random.choice

    Attributes
    ----------
        val_sets: list of tuples of strings
            1st level sublist correspond to a validation set; 2nd level sublist in the form of \
            ('sample_id', 'partition_id') for each dataset selected for the validation set
        errors: list


        best_params: dictionary
            key as the name of the parameter and value as the best scoring value the parameter can take

    Raises
        ------
            NotImplementedError
                only support tuning 1 parameter; hme.messi or single-response models (e.g. sklearn)


    """

    def __init__(self, params_dict, model, ratio, n_sets, params_list=None, multiclass=False,
                 output_file='./val.pickle', random_state=222):
        # parameters
        self.params_dict = params_dict

        if params_list is not None:
            self.params_list = params_list
        else:
            # generate all combinations of the parameters to be tuned
            keys, values = zip(*params_dict.items())
            self.params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

        self.ratio = ratio
        self.model = model
        self.multiclass = multiclass
        self.n_sets = n_sets
        self.output_file = output_file
        self.random_state = random_state

        # attributes
        self.val_sets = []
        self.errors = []
        self.best_params = {}

        # initialize errors
        for i in range(len(self.params_list)):
            self.errors.append([])


    def generate_val_sets(self, samples, meta_per_dataset):
        """

        Generate all validation sets represented by the meta data by random sampling from the full sets of
        datasets based on the set ratio. Specifically, the validations datasets are selected randomly without
        replacement for each sample separately applying the same ratio.

        Parameters
        ----------
            samples: list
                list of animal/sample IDs, corresponding to 'animal_id' in meta_per_dataset
            meta_per_dataset: list of tuples of strings
                each sublist in the form of ('sample_id', 'partition_id') for each dataset

        """
        # get available datasets for each animal/sample as dictionary
        aval_sets = {}
        for _id, _data in meta_per_dataset:
            if _id not in aval_sets:
                aval_sets[_id] = [[_id, _data]]
            else:
                aval_sets[_id].append([_id, _data])

        rng = np.random.RandomState(self.random_state)
        # for each set of validation data
        for i in range(self.n_sets):
            cur_set = []
            # for each animal/sample, random select a sets of validation sets based on the ratio
            for j in range(len(samples)):
                _id = samples[j]

                # get all available datasets for this sample
                all_aval = aval_sets[_id]

                # get number of validatation sets to select based on the ratio
                n_select = math.ceil(self.ratio * len(all_aval))  # some animal/sample will end up with no val

                # randomly select validation sets
                _selected = rng.choice(range(len(all_aval)), n_select, replace=False)

                # append to the current validation set
                for j in _selected:
                    cur_set.append(tuple(all_aval[j]))

            self.val_sets.append(cur_set)

    @staticmethod
    def get_idx_in_full(meta, meta_idx, data_type):
        """

        Get the indexes of validation / training samples.

        Parameters
        ----------
            meta: list of tuples of strings
                meta data for a single validation/train set; each sublist in the form of ('sample_id', 'partition_id') \
                for each dataset selected for validation/train
            meta_idx: dictionary
                key as ('sample_id', 'partition_id'), value as the start and end index + 1 of the samples in the \
                full data (merfish); or as the list of indexes (starmap or merfish_cell_line)
            data_type: string
                merfish/starmap/merfish_cell_line
        Returns
        -------
             idx_in_full: list
                list of indexes of validation/train samples as in the full data

        """

        idx_in_full = []
        for _data in meta:
            if data_type == 'merfish':
                cur_range = meta_idx[_data]
                idx_in_full += list(range(cur_range[0], cur_range[1]))
            else:
                idx_in_full += meta_idx[_data]

        return idx_in_full

    def runCV(self, Xs, Y, meta_per_dataset, meta_idx, feature_types, data_type, preprocess):

        """
        Run cross-validation for each candidate value (or values combination) of the parameter to be
        tuned.

        Parameters
        ----------

            Xs: dictionary of number array
                each item for a type of features; sample size x feature dimension
            Y: numpy array
                sample size x number of responses; response variables;
            meta_per_dataset: list of tuples of strings
                meta data for a single validation/train set; each sublist in the form of ('sample_id', 'partition_id') \
                for each dataset selected for validation/train
            meta_idx: dictionary
                key as ('sample_id', 'partition_id'), value as the start and end index + 1 of the samples in the full data
            feature_types: dictionary
                each specifying parameters of a particular feature type
            data_type: string
                merfish/starmap/merfish_cell_line
            preprocess: string
                the way to include neighborhood information; neighbor_cat: include by concatenating them to
                the cell own features; neighbor_sum: include by addinding to the cell own features; anything
                without 'neighbor': no neighborhood information will be used as features; 'baseline': only
                baseline features;

        """

        # for each validation sets
        for i in range(len(self.val_sets)):
            meta_val = self.val_sets[i]
            meta_train = [i for i in meta_per_dataset if i not in meta_val]

            # get index
            idx_val = self.get_idx_in_full(meta_val, meta_idx, data_type)
            idx_train = self.get_idx_in_full(meta_train, meta_idx, data_type)

            # subset the features for train and validation sets
            X_trains = copy.deepcopy(Xs)
            X_tests = copy.deepcopy(Xs)

            for j in range(0, len(feature_types)):
                feature_type = feature_types[j]
                feature_name = feature_type['name']

                X_trains[feature_name] = X_trains[feature_name][idx_train, :]
                X_tests[feature_name] = X_tests[feature_name][idx_val, :]

            # transformation
            transform_features(X_trains, X_tests, feature_types)

            # combine different type of features
            if data_type == 'merfish':
                num_coordinates = 3
            elif data_type == 'starmap' or data_type == 'merfish_cell_line':
                num_coordinates = 2
            else:
                num_coordinates = None

            if np.ndim(X_trains['baseline']) > 1 and np.ndim(X_tests['baseline']) > 1:
                X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)
                X_test, X_test_clf_1, X_test_clf_2 = combine_features(X_tests, preprocess, num_coordinates)
            elif np.ndim(X_trains['baseline']) > 1:
                X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)

            # subset the responses for train and validation sets
            Y_train = Y[idx_train, :]
            Y_test = Y[idx_val, :]

            # run CV for each combination of parameter values
            for idx_param in range(len(self.params_list)):
                cur_params = self.params_list[idx_param]
                print(f"Now running at {cur_params}")

                if type(self.model).__name__ == 'hme':

                    # train model

                    if 'n_classes_0' in cur_params:
                        raise NotImplementedError(f"Now only support tuning parameter for 1 layer HME for messi!")

                    # redo initialization
                    if 'n_classes_1' in cur_params:
                        # ------ initialize the sample assignments -------
                        model = AgglomerativeClustering(n_clusters=cur_params['n_classes_1'])
                        model = model.fit(Y_train)
                        hier_labels = [model.labels_]

                        # update model with the new parameter setting
                        new_params = copy.copy(cur_params)
                        new_params["init_labels_1"] = hier_labels
                    else:
                        # update model with the new parameter setting
                        new_params = copy.copy(cur_params)

                    self.model.set_params(**new_params)
                    self.model.initialize_gates()
                    self.model.initialize_experts()

                    print(f"Number of experts for current model is {len(self.model.model_experts[0])} \n"
                          f"Soft weight is {self.model.soft_weights}")

                    # ------ construct MESSI  ------
                    # train
                    self.model.train(X_train, X_train_clf_1, X_train_clf_2, Y_train)

                    # test model
                    Y_hat_final = self.model.predict(X_test, X_test_clf_1, X_test_clf_2)

                    mae = abs(Y_test - Y_hat_final).mean(axis=1).mean()
                    self.errors[idx_param].append(mae)

                elif not self.multiclass:

                    # update model with the new parameter setting
                    new_params = copy.copy(cur_params)
                    self.model.set_params(**new_params)
                    print(f"Model parameters after updating: {self.model.get_params()} ")

                    # train single gene at a time
                    Y_hat_final = []
                    for _g in range(0, Y_train.shape[1]):
                        # ------ current response ------
                        y_train = Y_train[:, _g]

                        self.model.fit(X_train_clf_2, y_train)  # _clf_2 contains all available features
                        y_hat = self.model.predict(X_test_clf_2)
                        Y_hat_final.append(y_hat)

                    Y_hat_final = np.array(Y_hat_final).T
                    mae = abs(Y_test - Y_hat_final).mean(axis=1).mean()
                    self.errors[idx_param].append(mae)

                else:
                    raise NotImplementedError(f"Now only support messi.hme or single-class models!")


    def get_best_parameter(self):
        """

        Select the best parameter (or combination) giving the best score based on majority vote. If tied, then
        prefer the ne with a smaller value.


        """
        array = np.array(self.errors)
        best_per_run = np.argmin(array, axis=0)
        mode_result = stats.mode(best_per_run)  # if tied, smaller ones will be selected
        self.best_params = self.params_list[mode_result[0][0]]

