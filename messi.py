#!/usr/bin/env python

"""
Train and test MESSI model on an expression dataset.

This script can either conduct cross-validation (when -mode = 'CV') where it goes through different
CV set (each with an animal/sample left out), processing the features and responses, constructing and training
a MESSI model and finally saving the learned model and the prediction results. When '-mode' = 'train',
It use all available samples to train and save the learned model. When '-grid_search' = True, grid search on
'n_classes_1' will be conducted using the training data. The best value will be used to refit the whole train data.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the script

"""

import sys
import os
import argparse
import pickle
import copy
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from context import messi
from messi.data_processing import *
from messi.hme import hme
from messi.gridSearch import gridSearch
from messi.utils import *

# search range for the number of experts if conduct grid search
search_range_dict = {'Inhibitory': range(7, 11), 'Excitatory': range(7, 11), 'Astrocyte': range(3, 7), \
                         'OD Mature 2': range(3, 7), 'Endothelial 1': range(1, 5), \
                         'OD Immature 1': range(1, 5), 'OD Mature 1': range(1, 5), \
                         'Microglia': range(1, 5), 'U-2_OS': range(1,5), \
                        'STARmap_excitatory': range(1,5)}


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        default='input/', help="string, path to the input folder with the expression data, "
                                                  "default 'input/'")
    parser.add_argument('-o', '--output', required=True,
                        default='output/', help="string, path to the output folder, default 'output/'")
    parser.add_argument('-d', '--dataType', required=True,
                        default='merfish', choices=['merfish', 'merfish_cell_line', 'starmap'],
                        help="string, type of expression data, 'merfish' for MERFISH hypothalamus data, "
                             "'merfish_cell_line' for MERFISH U-2 OS cells, 'starmap' for 'STARmap mPFC cells';"
                             "default 'merfish'")
    parser.add_argument('-g', '--gender', required=True,
                        default='Female', help="string, gender of input animal sample, default 'Female', put 'na' if "
                                               "not available")
    parser.add_argument('-b', '--behavior', required=True,
                        default='Naive', help="string, behavior of input animal sample, default 'Naive', put 'na' if "
                                              "not available")
    parser.add_argument('-c', '--cellType', required=True,
                        default='Excitatory', help="string, cell type that will be built a model for, "
                                                   "use \\ for white-space, e.g. 'OD\ Mature\ 2', default 'Excitatory'")
    parser.add_argument('-m', '--mode', required=True,
                        default='train', help="string, any of 'train', 'CV'; if 'train', then all data will be used "
                                              "for training and output a pickle file for learned parameters; if 'CV', "
                                              "then cross-validation will be conducted each time with an animal/sample "
                                              "left out and each CV run output a pickle file and prediction result, "
                                              "default 'train'")
    parser.add_argument('-c1', '--numLevel1', required=False,
                        default=1, help="integer, number of classes at level 1, number of experts = number of classes "
                                        "at level 1 x number of classes at level 2, default 1")
    parser.add_argument('-c2', '--numLevel2', required=False,
                        default=5, help="integer, number of classes at level 2, default 5")
    parser.add_argument('-e', '--epochs', required=False,
                        default=20, help="integer, number of epochs to train MESSI, default 20")
    parser.add_argument('-gs', '--grid_search', required=False, type=str2bool,
                        default=False, help="boolean, if conduct grid search for hyper-parameters, default False")
    parser.add_argument('-ns', '--n_sets', required=False,
                        default=3, help="integer, number of CV sets for grid search, default 3")
    parser.add_argument('-r', '--numReplicates', required=False,
                        default=1, help="integer, optional, number of times to run with same set of parameters, "
                                        "default 1")
    parser.add_argument('-p', '--preprocess', required=False,
                        default='neighbor_cat', help="string, optional, the way to include neighborhood information; "
                                                     "neighbor_cat: include by concatenating them to the cell own "
                                                     "features; neighbor_sum: include by addinding to the cell own "
                                                     "features; anything without 'neighbor': no neighborhood "
                                                     "information will be used as features; 'baseline': only baseline "
                                                     "features; default 'neighbor_cat'")

    parser.add_argument('-tr', '--topKResponses', required=False,
                        default=None, help='integer, optional, number of top dispersed responses genes to model,'
                                           'default None (to include all response genes)')
    parser.add_argument('-ts', '--topKSignals', required=False,
                        default=None, help='integer, optional, number of top dispersed signalling genes to use as '
                                           'features, default None (to include all signalling genes)')
    # parser.add_argument('-rp', '--responsePrior', required=False,
    #                     default=None, help='string, optional, path to the response genes to be used, default None')
    # parser.add_argument('-sp', '--signalsPrior', required=False,
    #                     default=None, help='string, optional, path to the signalling genes to be used, default None')

    args = parser.parse_args()
    print(args)

    # set parameters for data
    input_path = args.input
    output_path = args.output
    data_type = args.dataType
    sex = args.gender
    behavior = args.behavior
    behavior_no_space = behavior.replace(" ", "_")
    current_cell_type = args.cellType
    current_cell_type_no_space = current_cell_type.replace(" ", "_")

    # set parameters for model
    mode = args.mode
    grid_search = args.grid_search
    n_sets = int(args.n_sets)
    n_classes_0 = int(args.numLevel1)-
    n_classes_1 = int(args.numLevel2)
    n_epochs = int(args.epochs)
    n_replicates = int(args.numReplicates)

    # set parameters for data processing
    preprocess = args.preprocess
    if args.topKResponses is not None:
        top_k_response = int(args.topKResponses)
    else:
        top_k_response = args.topKResponses
    if args.topKSignals is not None:
        top_k_regulator  = int(args.topKSignals)
    else:
        top_k_regulator = args.topKSignals

    response_type = 'original'  # use raw values to fit the model

    if grid_search:
        condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_grid_search"
    else:
        condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_{n_classes_1}"

    # prepare to read data
    read_in_functions = {'merfish': [read_meta_merfish, read_merfish_data, get_idx_per_dataset_merfish],
                         'merfish_cell_line': [read_meta_merfish_cell_line, read_merfish_cell_line_data,
                                               get_idx_per_dataset_merfish_cell_line],
                         'starmap': [read_meta_starmap_combinatorial, read_starmap_combinatorial,
                                     get_idx_per_dataset_starmap_combinatorial]}

    # set data reading functions corresponding to the data type
    if data_type in ['merfish', 'merfish_cell_line', 'starmap']:
        read_meta = read_in_functions[data_type][0]
        read_data = read_in_functions[data_type][1]
        get_idx_per_dataset = read_in_functions[data_type][2]
    else:
        raise NotImplementedError(f"Now only support processing 'merfish', 'merfish_cell_line' or 'starmap'")

    # read in ligand and receptor lists
    l_u, r_u = get_lr_pairs()  # may need to change to the default value

    # read in meta information about the dataset
    meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u, \
    response_list_prior, regulator_list_prior = \
        read_meta(input_path, behavior_no_space, sex, l_u, r_u)  # TO BE MODIFIED: number of responses

    # get all available animals/samples
    all_animals = list(set(meta_all[:, meta_all_columns['Animal_ID']]))

    for _z in range(len(all_animals)):
        if mode == 'train':
            # only run once
            if _z == 0:
                test_animal = ''
            else:
                break
        else:
            test_animal = all_animals[_z]

        samples_test = np.array([test_animal])
        samples_train = np.array(list(set(all_animals) - {test_animal}))
        print(f"Test set is {samples_test}")
        print(f"Training set is {samples_train}")

        bregma = None
        # ------ read data ------
        idx_train, idx_test, idx_train_in_general, \
        idx_test_in_general, idx_train_in_dataset, \
        idx_test_in_dataset, meta_per_dataset_train, \
        meta_per_dataset_test = find_idx_for_train_test(samples_train, samples_test,
                                                        meta_all, meta_all_columns,
                                                        data_type, current_cell_type, get_idx_per_dataset,
                                                        return_in_general=False, bregma=bregma)

        # TBD: the current approach uses a lot memory;
        data_sets = []

        for animal_id, bregma in meta_per_dataset_train:
            hp, hp_cor, hp_genes = read_data(input_path, bregma, animal_id, genes_list, genes_list_u)

            if hp is not None:
                hp_columns = dict(zip(hp.columns, range(0, len(hp.columns))))
                hp_np = hp.to_numpy()
            else:
                hp_columns = None
                hp_np = None
            hp_cor_columns = dict(zip(hp_cor.columns, range(0, len(hp_cor.columns))))
            hp_genes_columns = dict(zip(hp_genes.columns, range(0, len(hp_genes.columns))))
            data_sets.append([hp_np, hp_columns, hp_cor.to_numpy(), hp_cor_columns,
                              hp_genes.to_numpy(), hp_genes_columns])
            del hp, hp_cor, hp_genes

        datasets_train = data_sets

        data_sets = []

        for animal_id, bregma in meta_per_dataset_test:
            hp, hp_cor, hp_genes = read_data(input_path, bregma, animal_id, genes_list, genes_list_u)

            if hp is not None:
                hp_columns = dict(zip(hp.columns, range(0, len(hp.columns))))
                hp_np = hp.to_numpy()
            else:
                hp_columns = None
                hp_np = None

            hp_cor_columns = dict(zip(hp_cor.columns, range(0, len(hp_cor.columns))))
            hp_genes_columns = dict(zip(hp_genes.columns, range(0, len(hp_genes.columns))))
            data_sets.append([hp_np, hp_columns, hp_cor.to_numpy(), hp_cor_columns,
                              hp_genes.to_numpy(), hp_genes_columns])
            del hp, hp_cor, hp_genes

        datasets_test = data_sets

        del data_sets

        # ------ pre-processing -------

        # construct neighborhood graph
        if data_type == 'merfish_RNA_seq':
            neighbors_train = None
            neighbors_test = None
        else:
            if data_type == 'merfish':
                dis_filter = 100
            else:
                dis_filter = 1e9

            neighbors_train = get_neighbors_datasets(datasets_train, "Del", k=10, dis_filter=dis_filter,
                                                     include_self=False)
            neighbors_test = get_neighbors_datasets(datasets_test, "Del", k=10, dis_filter=dis_filter,
                                                    include_self=False)
        # set parameters for different feature types
        lig_n = {'name': 'regulators_neighbor', 'helper': preprocess_X_neighbor_per_cell,
                 'feature_list_type': 'regulator_neighbor', 'per_cell': True, 'baseline': False,
                 'standardize': True, 'log': True, 'poly': False}
        rec_s = {'name': 'regulators_self', 'helper': preprocess_X_self_per_cell,
                 'feature_list_type': 'regulator_self', 'per_cell': True, 'baseline': False,
                 'standardize': True, 'log': True, 'poly': False}
        lig_s = {'name': 'regulators_neighbor_self', 'helper': preprocess_X_self_per_cell,
                 'feature_list_type': 'regulator_neighbor', 'per_cell': True, 'baseline': False,
                 'standardize': True, 'log': True, 'poly': False}
        type_n = {'name': 'neighbor_type', 'helper': preprocess_X_neighbor_type_per_dataset,
                  'feature_list_type': None, 'per_cell': False, 'baseline': False,
                  'standardize': True, 'log': False, 'poly': False}
        base_s = {'name': 'baseline', 'helper': preprocess_X_baseline_per_dataset, 'feature_list_type': None,
                  'per_cell': False, 'baseline': True, 'standardize': True, 'log': False, 'poly': False}

        if data_type == 'merfish_cell_line':
            feature_types = [lig_n, rec_s, base_s, lig_s]
        else:
            feature_types = [lig_n, rec_s, type_n, base_s, lig_s]

        # untransformed features
        X_trains, X_tests, regulator_list_neighbor, regulator_list_self = prepare_features(data_type, datasets_train,
                                                                                           datasets_test,
                                                                                           meta_per_dataset_train,
                                                                                           meta_per_dataset_test,
                                                                                           idx_train, idx_test,
                                                                                           idx_train_in_dataset,
                                                                                           idx_test_in_dataset,
                                                                                           neighbors_train,
                                                                                           neighbors_test,
                                                                                           feature_types,
                                                                                           regulator_list_prior,
                                                                                           top_k_regulator,
                                                                                           genes_list_u, l_u, r_u,
                                                                                           cell_types_dict)
        total_regulators = regulator_list_neighbor + regulator_list_self

        log_response = True  # take log transformation of the response genes
        Y_train, Y_train_true, Y_test, Y_test_true, response_list = prepare_responses(data_type, datasets_train,
                                                                                      datasets_test,
                                                                                      idx_train_in_general,
                                                                                      idx_test_in_general,
                                                                                      idx_train_in_dataset,
                                                                                      idx_test_in_dataset,
                                                                                      neighbors_train,
                                                                                      neighbors_test,
                                                                                      response_type, log_response,
                                                                                      response_list_prior,
                                                                                      top_k_response,
                                                                                      genes_list_u, l_u, r_u)
        if grid_search:
            X_trains_gs = copy.deepcopy(X_trains)
            Y_train_gs = copy.copy(Y_train)

        # transform features
        transform_features(X_trains, X_tests, feature_types)
        print(f"Minimum value after transformation can below 0: {np.min(X_trains['regulators_self'])}")

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

        print(f"Dimension of X train is: {X_train.shape}")
        if mode == 'CV':
            print(f"Dimension of X test is: {X_test.shape}")

        # ------ modeling by MESSI ------
        for _i in range(0, n_replicates):

            # ------ set parameters ------
            model_name_gates = 'logistic'
            model_name_experts = 'mrots'

            soft_weights = True
            partial_fit_expert = True

            # specify default parameters for MESSI
            model_params = {'n_classes_0': n_classes_0,
                            'n_classes_1': n_classes_1,
                            'model_name_gates': model_name_gates,
                            'model_name_experts': model_name_experts,
                            'num_responses': Y_train.shape[1],
                            'soft_weights': soft_weights,
                            'partial_fit_expert': partial_fit_expert,
                            'n_epochs': n_epochs,
                            'tolerance': 3}

            print(f"Model parameters for training is {model_params}")

            # set up directory for saving the model
            sub_condition = f"{condition}_{model_name_gates}_{model_name_experts}"
            sub_dir = f"{data_type}/{behavior_no_space}/{sex}/{current_cell_type_no_space}/{preprocess}/{sub_condition}"
            current_dir = os.path.join(output_path, sub_dir)

            if not os.path.exists(current_dir):
                os.makedirs(current_dir)

            print(f"Model and validation results (if applicable) saved to: {current_dir}")

            if mode == 'CV':
                suffix = f"_{test_animal}_{_i}"
            else:
                suffix = f"_{_i}"

            if grid_search:
                # prepare input meta data
                if data_type == 'merfish':
                    meta_per_part = [tuple(i) for i in meta_per_dataset_train]
                    meta_idx = meta2idx(idx_train_in_dataset, meta_per_part)
                else:
                    meta_per_part, meta_idx = combineParts(samples_train, datasets_train, idx_train_in_dataset)

                # prepare parameters list to be tuned
                if data_type == 'merfish_cell_line':
                    current_cell_type_data = 'U-2_OS'
                elif data_type == 'starmap':
                    current_cell_type_data = 'STARmap_excitatory'
                else:
                    current_cell_type_data = current_cell_type

                params = {'n_classes_1': list(search_range_dict[current_cell_type_data]), 'soft_weights': [True, False],
                          'partial_fit_expert': [True, False]}

                keys, values = zip(*params.items())
                params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

                new_params_list = []
                for d in params_list:
                    if d['n_classes_1'] == 1:
                        if d['soft_weights'] and d['partial_fit_expert']:
                            # n_expert = 1, soft or hard are equivalent
                            new_params_list.append(d)
                    else:
                        if d['soft_weights'] == d['partial_fit_expert']:
                            new_params_list.append(d)
                ratio = 0.2

                # initialize with default values
                model_params_val = model_params.copy()
                model_params_val['n_epochs'] = 5  # increase for validation models to converge
                model_params_val['tolerance'] = 0
                print(f"Default model parameters for validation {model_params_val}")
                model = hme(**model_params_val)

                gs = gridSearch(params, model, ratio, n_sets, new_params_list)
                gs.generate_val_sets(samples_train, meta_per_part)
                gs.runCV(X_trains_gs, Y_train_gs, meta_per_part, meta_idx, feature_types, data_type,
                         preprocess)
                gs.get_best_parameter()
                print(f"Best params from grid search: {gs.best_params}")

                # modify the parameter setting
                for key, value in gs.best_params.items():
                    model_params[key] = value

                print(f"Model parameters for training after grid search {model_params}")

                filename = f"validation_results{suffix}.pickle"
                pickle.dump(gs, open(os.path.join(current_dir, filename), 'wb'))

            # ------ initialize the sample assignments ------

            if grid_search and 'n_classes_1' in params:
                model = AgglomerativeClustering(n_clusters=gs.best_params['n_classes_1'])
            else:
                model = AgglomerativeClustering(n_classes_1)

            model = model.fit(Y_train)
            hier_labels = [model.labels_]
            model_params['init_labels_1'] = hier_labels

            # ------ construct MESSI  ------
            model = hme(**model_params)

            # train
            model.train(X_train, X_train_clf_1, X_train_clf_2, Y_train)
            if grid_search and 'n_classes_1' in params:
                model = AgglomerativeClustering(n_clusters=gs.best_params['n_classes_1'])
            else:
                model = AgglomerativeClustering(n_classes_1)

            model = model.fit(Y_train)
            hier_labels = [model.labels_]
            model_params['init_labels_1'] = hier_labels

            # ------ construct MESSI  ------
            model = hme(**model_params)

            # train
            model.train(X_train, X_train_clf_1, X_train_clf_2, Y_train)
            # save the model
            filename = f"hme_model{suffix}.pickle"
            pickle.dump(model, open(os.path.join(current_dir, filename), 'wb'))

            # predict the left-out animal
            if mode == 'CV':

                Y_hat_final = model.predict(X_test, X_test_clf_1, X_test_clf_2)

                mae = abs(Y_test - Y_hat_final).mean(axis=1).mean()
                print(f"Mean absolute value for {test_animal} is {mae}")

                filename = f"test_predictions_{test_animal}_{_i}"
                np.save(os.path.join(current_dir, filename), Y_hat_final)


if __name__ == '__main__':
    main()
