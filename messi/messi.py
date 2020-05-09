"""
Train and test MESSI model on an expression dataset.

This script can either conduct cross-validation (when -mode = 'CV') where it goes through different
CV set (eahc with an animal/sample left out), processing the features and responses, constructing and training
a MESSI model and finalling saving the learned model and the prediction results. When -mode = 'train',
It use all available samples to train and save the learned model.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the script

"""



import sys
import os
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from data_processing import *
from hme import hme


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        default='../input/', help="string, path to the input folder with the expression data, "
                                                  "default '../input/'")
    parser.add_argument('-o', '--output', required=True,
                        default='../output/', help="string, path to the output folder, default '../output/'")
    parser.add_argument('-d', '--dataType', required=True,
                        default='merfish', help="string, type of expression data, default 'merfish'")
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
    parser.add_argument('-c1', '--numLevel1', required=True,
                        default=1, help="integer, number of classes at level 1, number of experts = number of classes "
                                        "at level 1 x number of classes at level 2, default 1")
    parser.add_argument('-c2', '--numLevel2', required=True,
                        default=5, help="integer, number of classes at level 2, default 5")
    parser.add_argument('-e', '--epochs', required=False,
                        default=20, help="integer, number of epochs to train MESSI, default 20")
    parser.add_argument('-r', '--numReplicates', required=False,
                        default=1, help="integer, optional, number of times to run with same set of parameters, "
                                        "default 1")
    parser.add_argument('-p', '--preprocess', required=False,
                        default='neighbor_cat', help="string, optional, the way to include neighborhood information; "
                                                     "neighbor_cat: include by concatenating them to the cell own "
                                                     "features; neighbor_sum: include by addinding to the cell own "
                                                     "features; anything without 'neighbor': no neighborhood "
                                                     "information will be used as features, default 'neighbor_cat'")

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
    n_classes_0 = int(args.numLevel1)
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
    response_list_prior = None
    regulator_list_prior = None
    response_type = 'original'  # use raw values to fit the model
    condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_{n_classes_1}"

    # set data reading functions corresponding to the data type
    if data_type == 'merfish':
        read_meta = read_meta_merfish
        read_data = read_merfish_data
        get_idx_per_dataset = get_idx_per_dataset_merfish
    else:
        raise NotImplementedError(f"Now only support processing MERFISH hypothalamus data.")

    # read in ligand and receptor lists
    l_u, r_u = get_lr_pairs()

    # read in meta information about the dataset
    meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u = \
        read_meta(input_path, behavior_no_space, sex)

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
            try:
                hp, hp_cor, hp_genes = read_data(input_path, bregma, animal_id, genes_list)
            except TypeError:
                continue

            hp_columns = dict(zip(hp.columns, range(0, len(hp.columns))))
            hp_cor_columns = dict(zip(hp_cor.columns, range(0, len(hp_cor.columns))))
            hp_genes_columns = dict(zip(hp_genes.columns, range(0, len(hp_genes.columns))))
            data_sets.append([hp.to_numpy(), hp_columns, hp_cor.to_numpy(), hp_cor_columns,
                              hp_genes.to_numpy(), hp_genes_columns])
            del hp, hp_cor, hp_genes

        datasets_train = data_sets

        data_sets = []

        for animal_id, bregma in meta_per_dataset_test:
            try:
                hp, hp_cor, hp_genes = read_data(input_path, bregma, animal_id, genes_list)
            except TypeError:
                continue

            hp_columns = dict(zip(hp.columns, range(0, len(hp.columns))))
            hp_cor_columns = dict(zip(hp_cor.columns, range(0, len(hp_cor.columns))))
            hp_genes_columns = dict(zip(hp_genes.columns, range(0, len(hp_genes.columns))))
            data_sets.append([hp.to_numpy(), hp_columns, hp_cor.to_numpy(), hp_cor_columns,
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
        # combine different type of features
        if data_type == 'merfish':
            num_coordinates = 3
        elif data_type == 'starmap' or data_type == 'merfish_cell_line':
            num_coordinates = 2
        else:
            num_coordinates = None

        if np.ndim(X_trains['regulators_self']) > 1 and np.ndim(X_tests['regulators_self']) > 1:
            X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)
            X_test, X_test_clf_1, X_test_clf_2 = combine_features(X_tests, preprocess, num_coordinates)
        elif np.ndim(X_trains['regulators_self']) > 1:
            X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)

        print(f"Dimension of X train is: {X_train.shape}")
        if mode == 'CV':
            print(f"Dimension of X test is: {X_test.shape}")

        # ------ modeling by MESSI ------
        for _i in range(0, n_replicates):

            # ------ set parameters ------
            model_name_gates = 'logistic'
            model_name_experts = 'mrots'
            num_response = Y_train.shape[1]
            tolerance = 3

            if current_cell_type not in ['OD Mature 2', 'Astrocyte', 'Endothelial 1']:
                # soft weights
                sub_condition = f"{condition}_{model_name_gates}_{model_name_experts}_soft"
                soft_weights = True
                partial_fit_expert = True

            else:
                # hard weights
                sub_condition = f"{condition}_{model_name_gates}_{model_name_experts}_soft"
                soft_weights = False
                partial_fit_expert = False

            print(f"Parameters for MESSI: number of classes at level 1: {n_classes_0}, level 2: {n_classes_1}\n\
                  model for gates: {model_name_gates}, model for experts: {model_name_experts}\n\
                  if used soft weights: {soft_weights}")

            # ------ initialize the sample assignments ------
            model = AgglomerativeClustering(n_clusters=n_classes_1)
            model = model.fit(Y_train)
            hier_labels = [model.labels_]

            # ------ construct MESSI  ------
            model = hme(n_classes_0, n_classes_1, model_name_gates, model_name_experts, num_response,
                        init_labels_1=hier_labels, soft_weights=soft_weights, partial_fit_expert=partial_fit_expert,
                        n_epochs=n_epochs, tolerance=tolerance)
            # train
            model.train(X_train, X_train_clf_1, X_train_clf_2, Y_train)

            # save the model
            sub_dir = f"{data_type}/{behavior_no_space}/{sex}/{current_cell_type_no_space}/{preprocess}/{sub_condition}"
            current_dir = os.path.join(output_path, sub_dir)

            if not os.path.exists(current_dir):
                os.makedirs(current_dir)

            print(f"Save to: {current_dir}")

            if mode == 'CV':
                suffix = f"_{test_animal}_{_i}"
            else:
                suffix = f"_{_i}"

            filename = f"hme_model{suffix}.pickle"
            pickle.dump(model, open(os.path.join(current_dir, filename), 'wb'))

            # predict the left-out animal
            if mode == 'CV':

                Y_hat_final = model.predict(X_test, X_test_clf_1, X_test_clf_2)

                mae = abs(Y_test - Y_hat_final).sum(axis=1).mean()
                print(f"Mean absolute value for {test_animal} is {mae}")

                filename = f"test_predictions_{test_animal}_{_i}"
                np.save(os.path.join(current_dir, filename), Y_hat_final)


if __name__ == '__main__':
    main()
