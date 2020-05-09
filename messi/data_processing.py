"""
Functions for processing data for MESSI.

This includes functions to read helper data (e.g., ligand, receptor list), to read
various types of expression data and their meta data, to construct neighberhood graph and find
neighbors and to prepare features and response variables.

"""

import os

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

# ------ read data ------

# tool data
def get_lr_pairs(filename = 'ligand_receptor_pairs2.txt', input_path = 'input/'):
    """
    Process and read in the ligand and receptor list.

    Args:
        filename: string, filename of the ligand and receptor list
        input_path:

    Returns:
        l_u: set of ligands
        r_u: set of receptors

    """
    lr_pairs = pd.read_table(os.path.join(input_path, filename), header = None)
    lr_pairs.columns = ['ligand','receptor']
    lr_pairs['ligand'] = lr_pairs['ligand'].apply(lambda x: x.upper())
    lr_pairs['receptor'] = lr_pairs['receptor'].apply(lambda x: x.upper())
    l_u_p = set([l.upper() for l in lr_pairs['ligand']])
    r_u_p = set([g.upper() for g in lr_pairs['receptor']])

    l_u_search = set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
    r_u_search = set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])

    l_u = l_u_p.union(l_u_search)
    r_u = r_u_p.union(r_u_search)

    return l_u, r_u

# meta data
def read_meta_merfish(input_path, behavior_no_space, sex):
    """
    Get meta data of merfish hypothalamus datasets.

    Returns:
        meta_all: numpy array, sample size x meta features, meta data (e.g., animal ID or bregma) for a cell
        meta_all_columns: dictionary, use column name as key for column index
        cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
        genes_list: list of strings, all genes that are profiled in the dataset
        genes_list_u: list of strings, same as genes_list but all letters are in uppercase

    """

    # get default values (genes names, all cell types ) from a sample dataset
    filename = f"merfish_animal1_bregma026.csv"
    sample = pd.read_csv(os.path.join(input_path, filename), header=0)

    # remove columns for quality control
    genes_list = list(sample.columns[9:])
    remove = set(genes_list[12:17]).union(set(['Fos']))  # use Fos for validation only
    print(f"Removed genes: {remove}")

    for r in list(remove):
        genes_list.remove(r)

    genes_list_u = [g.upper() for g in genes_list]

    cell_types_dict = sample['Cell_class'].value_counts().to_dict()

    for k, v in cell_types_dict.items():
        cell_types_dict[k] = 0

    full_cell_types = list(cell_types_dict.keys())
    print(f"Total number of cell types for merfish: {len(full_cell_types)}")


    # get meta info for current samples
    filename = f"merfish_meta_{behavior_no_space}_{sex}.csv"
    meta_all = pd.read_csv(os.path.join(input_path, filename))
    meta_all_columns = dict(zip(meta_all.columns, range(0,len(meta_all.columns))))
    meta_all = meta_all.to_numpy()

    return meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u


# input data
# parameters for merfish
bregmas_all = [-0.29, -0.24, -0.19, -0.14, -0.09, -0.04, 0.01, 0.06, 0.11, 0.16, 0.21, 0.26]
female_ids = range(1,5)
male_ids = range(5,12)

def read_merfish_cell_line_data(input_path, bregma, animal_id, genes_list):
    """

    Read a subset of expression data by animal id and bregma (only for merfish
    hypothalamus).

    Returns:
        hp_cor: pandas DataFrame, spatial coordinates
        hp_genes: pandas DataFrame, gene expressions

    """
    try:
        filename = f"expression_cor_dispersion_6000_{animal_id}.csv"
        print(f"Reading file: {filename}")
        sample = pd.read_csv(os.path.join(input_path, filename), header=0)

        print(f"The dimensions of the sample is: {sample.shape}")

        hp = sample.copy()
        hp_cor = hp[['x_microns', 'y_microns']]
        print(hp_cor.shape)
        hp_genes = hp[genes_list]
        print(hp_genes.shape)
        hp_genes.columns = [g.upper() for g in hp_genes.columns]

        return None, hp_cor, hp_genes

    except FileNotFoundError:
        print(f"No file exist for this {bregma} of {animal_id}")

def read_merfish_data(input_path, bregma, animal_id, genes_list):
    """

    Read a subset of expression data by animal id and bregma (only for merfish
    hypothalamus).

    Returns:
        hp: pandas DataFrame, meta data
        hp_cor: pandas DataFrame, spatial coordinates
        hp_genes: pandas DataFrame, gene expressions

    """
    
    try:
        bregma_format = ''.join(str(bregma).split('.'))
        
        filename = f"merfish_animal{animal_id}_bregma{bregma_format}.csv"
        print(f"Reading file: {filename}")
        sample = pd.read_csv(os.path.join(input_path, filename), header = 0)

        print (f"The dimensions of the sample is: {sample.shape}")

        hp = sample.copy()
        hp_cor = hp[['Centroid_X','Centroid_Y']]
        hp_genes = hp[genes_list]
        hp_genes.columns = [g.upper() for g in hp_genes.columns]

        return hp.iloc[:, :9], hp_cor, hp_genes

    except FileNotFoundError:
        print(f"No file exist for this {bregma} of {animal_id}")
        
def read_starmap_combinatorial(input_path, animal_id, genes_list):

    """

    Read a subset of expression data by animal id and bregma (only for merfish
    hypothalamus).

    Returns:
        meta: pandas DataFrame, meta data
        cor: pandas DataFrame, spatial coordinates
        genes: pandas DataFrame, gene expressions

    """
    
    try:
        filename = f"meta_mPEC_{animal_id}.csv"
        print(f"Reading file: {filename}")
        meta = pd.read_csv(os.path.join(input_path, filename), header = 0)
        # meta = meta_all[meta_all['Animal_ID'] == animal_id]
        print (f"The dimensions of the meta is: {meta.shape}")

        filename = f"coordinates_{animal_id}.npy"
        print(f"Reading file: {filename}")
        cor = pd.DataFrame(np.load(os.path.join(input_path, filename)))
        cor.columns = ['Centroid_X','Centroid_Y']
        print (f"The dimensions of the cor is: {cor.shape}")
        
        filename = f"cell_barcode_count_{animal_id}.csv"
        print(f"Reading file: {filename}")
        genes = pd.read_csv(os.path.join(input_path, filename), header = None)
        genes.columns = genes_list
        print (f"The dimensions of the expression is: {genes.shape}")

        return meta, cor, genes
    except FileNotFoundError:
        print(f"No file exist for this {input_path} of {animal_id}")

        
        
def read_merfish_rna_seq_data(input_path, sex, animal_id, condition, genes_list=None):
    """

    Read a subset of expression data by animal id and bregma (only for merfish
    hypothalamus).

    Returns:
        hp: pandas DataFrame, meta data
        hp_genes: pandas DataFrame, gene expressions

    """
    try:
        filename = f"merfish_rna_seq_meta_{sex}_{animal_id}.csv"
        hp = pd.read_csv(os.path.join(input_path, filename), header = 0)
        print(f"Reading file: {filename}")

        if 'prior' in condition:
            filename = f"expression_{sex}_{animal_id}_prior.npy"
        else:
            filename = f"expression_{sex}_{animal_id}.npy"
        hp_genes = np.load(os.path.join(input_path, filename))
        print(f"Reading file: {filename}")
        print(f"The dimensions of the sample is: {hp_genes.shape}")

        return hp, None, hp_genes.T

    except FileNotFoundError:
        print(f"No file exist for this of {animal_id}")


# ------ data pre-processing 
# TBD:
# Now use a lot memory for datasets:
# 1. do not store all datasets! only need to process each one at a time 



# --- get index of train and test 
def find_idx_for_train_test(samples_train, samples_test, meta_all, 
                            meta_all_columns, data_type, current_cell_type, 
                            get_idx_per_dataset, return_in_general=True, bregma=None):
    """
    Get index of individual cells for given lists of a training and testing samples/animals.
    
    Args:
        samples_id_train: list, sample ID for training data; here sample might be 'animals'(merFISH) or 'FOV'(seqFISH+)
        samples_id_test: list, sample ID for testing data; here sample might be 'animals'(merFISH) or 'FOV'(seqFISH+)
        meta_all: numpy array, sample size x meta features, meta data (e.g., animal ID or bregma) for all cells of \
the same condition (say, all naive animals)
        meta_all_columns: dictionary, use column name as key for column index

    Returns:
        idx_train: list, index of training cells in meta_all 
        idx_test: list, index of testing cells in meta_all
        
        idx_train_in_general:list, index of training cells in the general (all cell types) list
        idx_test_in_general:list, index of testing cells in the general (all cell types) list
        
        idx_train_in_dataset: list of list of index of training cells in each dataset (as grouped by animal & bregma)
        idx_test_in_dataset: list of list of index of testing cells in each dataset (as grouped by animal & bregma)
        
        meta_per_dataset_train: list of list of ['animal_id', 'bregma'] for each dataset of training 
        meta_per_dataset_test: list of list of ['animal_id', 'bregma'] for each dataset of testing 

    """
    if data_type == 'seqfish':
    
        idx_train_general = []
        for fov in samples_train:
            idx_train_general = idx_train_general + list(meta_all.index[meta_all['Field of View'] == fov].values)

        idx_test_general = []
        for fov in samples_test:
            idx_test_general = idx_test_general + list(meta_all.index[meta_all['Field of View'] == fov].values)
            
        # for current_cell_type in cell_types:
        print(f"Preprocess for {current_cell_type} of {data_type}")

        # select cells for a particular cell type 
        if current_cell_type == 'general':
            idx_train = idx_train_general
            idx_test = idx_test_general

        else:
            idx_train = []
            for fov in samples_train:
                idx_train = idx_train + list(meta_all.index[(meta_all['Field of View'] == fov) & (meta_all['cell_types'] == current_cell_type)])

            idx_test = []
            for fov in samples_test:
                idx_test = idx_test + list(meta_all.index[(meta_all['Field of View'] == fov) & (meta_all['cell_types'] == current_cell_type)])

        idx_train_in_general = [idx_train_general.index(i) for i in idx_train]
        idx_test_in_general = [idx_test_general.index(i) for i in idx_test]
        
        print(len(idx_train))
        print(len(idx_test))
        
        return idx_train, idx_test, idx_train_in_general, idx_test_in_general, None, None, None, None
    
    elif 'merfish' in data_type or data_type == 'starmap': 
        
         # for current_cell_type in cell_types:
        print(f"Preprocess for {current_cell_type} of {data_type}")
        
        idx_train_general = []
        for animal in samples_train:
            idx_train_general = idx_train_general + list(np.where(meta_all[:, meta_all_columns['Animal_ID']] == animal)[0])

        idx_test_general = []
        for animal in samples_test:
            idx_test_general = idx_test_general + list(np.where(meta_all[:, meta_all_columns['Animal_ID']] == animal)[0])
        
        # select cells for a particular cell type  
        if current_cell_type == 'general':
        
            idx_train, idx_train_in_dataset, meta_per_dataset_train = \
            get_idx_per_dataset(samples_train,meta_all,meta_all_columns, bregma)
            idx_test, idx_test_in_dataset, meta_per_dataset_test = \
            get_idx_per_dataset(samples_test,meta_all, meta_all_columns, bregma)
        else:
            idx_train, idx_train_in_dataset, meta_per_dataset_train  = \
            get_idx_per_dataset(samples_train, meta_all, meta_all_columns, bregma, current_cell_type)
            idx_test, idx_test_in_dataset, meta_per_dataset_test  = \
            get_idx_per_dataset(samples_test, meta_all, meta_all_columns, bregma, current_cell_type)
        
        print(len(idx_train))
        print(len(idx_test))
        
        # cost MUCH time! TBD: concatenate baseline result --> 
        # no need to have idx_train_in_general for response_type == 'original'

        if return_in_general:
            idx_train_in_general = [idx_train_general.index(i) for i in idx_train]
            idx_test_in_general = [idx_test_general.index(i) for i in idx_test]

            return idx_train, idx_test, idx_train_in_general, idx_test_in_general, idx_train_in_dataset, \
                    idx_test_in_dataset, meta_per_dataset_train, meta_per_dataset_test
        else:
            return idx_train, idx_test, None, None, idx_train_in_dataset, \
                   idx_test_in_dataset, meta_per_dataset_train, meta_per_dataset_test

        
    else:
        print("Please provide a data type. Only 'seqfish','merfish' or 'starmap' is supported.")
        
        return None, None, None, None, None, None, None, None


def get_idx_per_dataset_merfish_cell_line(samples, meta_all, meta_all_columns,
                                        single_bregma=None, current_cell_type=None):
    """
    Helper function for find_idx_for_train_test.

    """

    idx_in_all = []
    idx_in_dataset = []
    meta_per_dataset = []

    for animal in samples:
        print(animal)

        _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal)

        #         _idx_in_all = list(meta_all.index[_condition].values)
        #         idx_in_all = idx_in_all + _idx_in_all

        _subidx = np.where(_condition)[0]
        _subset = meta_all[_condition, :]
        idx_in_all_animal = list(_subidx)
        idx_in_dataset.append(list(_subset[:, meta_all_columns['ID_in_dataset']]))
        meta_per_dataset.append([animal, None])

        idx_in_all = idx_in_all + idx_in_all_animal

    return idx_in_all, idx_in_dataset, meta_per_dataset

def get_idx_per_dataset_starmap_combinatorial(samples, meta_all, meta_all_columns, 
                                        single_bregma = None, current_cell_type = None):
    """

    Helper function for find_idx_for_train_test.

    """

    idx_in_all = []
    idx_in_dataset = []
    meta_per_dataset = []  

    for animal in samples:
        print(animal)

        if current_cell_type is not None:
            _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal) & \
            (meta_all[:, meta_all_columns['Cell_class']] == current_cell_type)
        else:
            _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal)

    #         _idx_in_all = list(meta_all.index[_condition].values)
    #         idx_in_all = idx_in_all + _idx_in_all

        _subidx = np.where(_condition)[0]
        _subset = meta_all[_condition,:]
        idx_in_all_animal = list(_subidx)
        idx_in_dataset.append(list(_subset[:,meta_all_columns['ID_in_dataset']]))
        meta_per_dataset.append([animal, None])

        idx_in_all = idx_in_all + idx_in_all_animal 

    return idx_in_all, idx_in_dataset, meta_per_dataset

# get_idx_per_dataset_merfish_rna_seq = get_idx_per_dataset_starmap_combinatorial()


def get_idx_per_dataset_merfish(samples, meta_all, meta_all_columns, single_bregma = None, current_cell_type = None):
    """

    Helper function for find_idx_for_train_test.

    """

    idx_in_all = []
    idx_in_dataset = []
    meta_per_dataset = []  

    for animal in samples:

        if current_cell_type is not None:
            _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal) & \
            (meta_all[:, meta_all_columns['Cell_class']] == current_cell_type)
        else:
            _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal)

    #         _idx_in_all = list(meta_all.index[_condition].values)
    #         idx_in_all = idx_in_all + _idx_in_all
        if single_bregma is not None:
            uniques = [single_bregma]
        else:
            uniques = meta_all[:, meta_all_columns['Bregma']][_condition]

        _bregmas = [b for b in bregmas_all if b in uniques]
        idx_in_all_animal = []

        for bregma in _bregmas:

            if current_cell_type is not None:
                _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal) & \
                (meta_all[:, meta_all_columns['Cell_class']] == current_cell_type) & \
                (meta_all[:, meta_all_columns['Bregma']] == bregma)
            else:
                _condition = (meta_all[:, meta_all_columns['Animal_ID']] == animal) & \
                (meta_all[:, meta_all_columns['Bregma']] == bregma)

            _subidx = np.where(_condition)[0]
            _subset = meta_all[_condition,:]
            idx_in_all_animal = idx_in_all_animal + list(_subidx)
            idx_in_dataset.append(list(_subset[:,meta_all_columns['ID_in_dataset']]))
            meta_per_dataset.append([animal,bregma])

        idx_in_all = idx_in_all + idx_in_all_animal 

    return idx_in_all, idx_in_dataset, meta_per_dataset


# --- find neighbors

def find_neighbors(pindex, tri):
    """
    By @user2535797 at https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-\
    in-a-delaunay-triangulation-using-sci

    Args:
        pindex: a index of a single point
        tri: output from scipy.Delaunay() method

    Returns:
        neighbors for a data point

    """
    return tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]]


def get_Delaunay_neighbors(X, dis_filter = 100, include_self = True):
    """
    Get neighbors based on Delaunay triangulation and physical distance filters.

    Args:
        X: np.array or pd.DataFrame, first dimension as data points and 2nd dimension as features
        include_self:boolean, if include self in neighbors
        dis_filter: integer, distance in micrometers to filter neighbor pairs; default 100
        include_self: boolean, if to include the point itself as its neighbors; default true

     Return:
        neighbors_filtered: list, neighbors filtered by euclidean distance ordered as in X
    """
    tri = Delaunay(X)
    distmat = squareform(pdist(X, 'euclidean'))
    neighbors_filtered = []
    
    for i in range(0, X.shape[0]):
        neighbors = find_neighbors(i,tri)
        temp = neighbors[distmat[i,neighbors] < dis_filter]
        
        if include_self:
            np.append(temp,i)

        neighbors_filtered.append(temp)
        
    return neighbors_filtered

def get_neighbors_datasets(data_sets, neighbor_type, k=10, dis_filter=100, include_self = True):
    """

    Process each dataset one by one to get neighbors for each single cell.

    Returns:
        neighbors_all: list of numpy arrays, each containing neighboring info of a dataset

    """
    
    neighbors_all = []
    
    for i in range(len(data_sets)):
        _, _, current_cor, _, _,_ = data_sets[i]
        
        if neighbor_type == 'Del':
            _nn = get_Delaunay_neighbors(current_cor, dis_filter=dis_filter, include_self=include_self)

        elif neighbor_type == 'k-nn':
            _nn_g = kneighbors_graph(current_cor, k=k, mode='connectivity', include_self=include_self)
            _nn = _nn_g.toarray()
        
        neighbors_all.append(_nn)
        
    return neighbors_all


# --- process X 

def prepare_features(data_type, datasets_train, datasets_test, meta_per_dataset_train, meta_per_dataset_test, idx_train,
                     idx_test, idx_train_in_dataset, idx_test_in_dataset,
                     neighbors_train, neighbors_test,
                     feature_types, regulator_list_prior, top_k_regulator,
                     genes_filtered, l_u, r_u, cell_types_dict):
    """

    Generate different types of features.

    Args:
        data_type: string, merfish/starmap/merfish_cell_line
        datasets_train: list of numpy arrays & dictionaries; all info for training samples;in the format
            [current_meta, gene_exp, current_cor]
        datasets_test: list of numpy arrays & dictionaries; all info for testing samples;in the format
            [current_meta, gene_exp, current_cor]
        meta_per_dataset_train: list of list of ['animal_id', 'bregma'] for each dataset of training
        meta_per_dataset_test: list of list of ['animal_id', 'bregma'] for each dataset of testing
        idx_train: list, index of training cells in the meta dataset containing all samples for a particular behavior
        idx_test: list, index of testing cells in the meta dataset containing all samples for a particular behavior
        idx_train_in_dataset: list of list of index of training cells in each dataset (as grouped by animal & bregma)
        idx_test_in_dataset: list of list of index of testing cells in each dataset (as grouped by animal & bregma)
        neighbors_train: list of numpy arrays, neighboring info for the training samples
        neighbors_test: list of numpy arrays, neighboring info for the testing samples
        feature_types: dictionary, each specifying parameters of a particular feature type
        regulator_list_prior: list of strings, ligand or receptor list known as prior
        top_k_regulator: integer, top K ligand/receptors to be used as features; the features will be sorted \
        according to dispersion (descending); default None
        genes_filtered: list of strings, all genes profiled in the expression data; all letters are in uppercase
        l_u: list of strings, ligands from the database; all letters are in uppercase
        r_u: list of strings, receptors from the database; all letters are in uppercase
        cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types

    Returns:
        X_trains: dictionary, each item for a type of features
        X_tests: dictionary, each item for a type of features
        regulator_list_neighbor: list of strings, ligands used as features for neighboring cells' expression
        regulator_list_self: list of strings, ligands and receptors used as features for cell's own expression

    """

    if regulator_list_prior is not None:
        regulator_list_neighbor = [g for g in regulator_list_prior if g in l_u]
        regulator_list_self = [g for g in regulator_list_prior if g in r_u]
        total_regulators = regulator_list_neighbor + regulator_list_self
    else:
        regulator_list_neighbor = [g for g in genes_filtered if g in l_u]
        regulator_list_self = [g for g in genes_filtered if g in r_u]
        total_regulators = regulator_list_neighbor + regulator_list_self

    # initialize
    X_trains = {'regulators_neighbor': None, 'regulators_self': None, 'neighbor_type': None, 'baseline': None,
                'regulators_neighbor_self': None, 'regulators_neighbor_by_type': None}
    X_tests = {'regulators_neighbor': None, 'regulators_self': None, 'neighbor_type': None, 'baseline': None,
               'regulators_neighbor_self': None, 'regulators_neighbor_by_type': None}

    for i in range(0, len(feature_types)):
        feature_type = feature_types[i]
        feature_name = feature_type['name']
        print(f"Now prepare for feature: {feature_name}")

        preprocess_X = feature_type['helper']

        feature_list_type = feature_type['feature_list_type']
        if feature_list_type == 'regulator_neighbor':
            feature_list = regulator_list_neighbor
        elif feature_list_type == 'regulator_self':
            feature_list = regulator_list_self
        else:
            feature_list = None

        per_cell = feature_type['per_cell']
        baseline = feature_type['baseline']

        data_sets = datasets_train
        meta_per_dataset = meta_per_dataset_train
        current_idxs = idx_train_in_dataset
        current_nns = neighbors_train

        X_train = preprocess_concatenate_X_datasets(data_sets, meta_per_dataset, data_type, current_idxs,
                                                    current_nns, feature_list, preprocess_X, cell_types_dict,
                                                    neighbor_type='Del', log=False, per_cell=per_cell,
                                                    baseline=baseline, poly=False)

        data_sets = datasets_test
        meta_per_dataset = meta_per_dataset_test
        current_idxs = idx_test_in_dataset
        current_nns = neighbors_test

        X_test = preprocess_concatenate_X_datasets(data_sets, meta_per_dataset, data_type, current_idxs,
                                                   current_nns, feature_list, preprocess_X, cell_types_dict,
                                                   neighbor_type='Del', log=False, per_cell=per_cell,
                                                   baseline=baseline, poly=False)

        assert X_train.shape[0] == len(idx_train)
        assert X_test.shape[0] == len(idx_test)

        X_trains[feature_name] = X_train
        X_tests[feature_name] = X_test

    # filter features based on original scale
    if top_k_regulator is not None:
        # filter regulators neighbor
        dispersion = X_trains['regulators_neighbor_self'].var(axis=0) / (
                    X_trains['regulators_neighbor_self'].mean(axis=0) + 1e-6)
        idx_filtered = np.argsort(-dispersion, axis=0)[:top_k_regulator]
        for key in ['regulators_neighbor', 'regulators_neighbor_self']:
            if X_trains[key] is not None:
                X_trains[key] = X_trains[key][:, idx_filtered]
                X_tests[key] = X_tests[key][:, idx_filtered]
                regulator_list_neighbor = list(np.array(regulator_list_neighbor)[idx_filtered])

        # filter regulators self
        dispersion = X_trains['regulators_self'].var(axis=0) / (X_trains['regulators_self'].mean(axis=0) + 1e-6)
        idx_filtered = np.argsort(-dispersion, axis=0)[:top_k_regulator]
        X_trains['regulators_self'] = X_trains['regulators_self'][:, idx_filtered]
        X_tests['regulators_self'] = X_tests['regulators_self'][:, idx_filtered]
        regulator_list_self = list(np.array(regulator_list_self)[idx_filtered])

        total_regulators = regulator_list_neighbor + regulator_list_self

    # transform features: log + standardize
    for i in range(0, len(feature_types)):
        feature_type = feature_types[i]
        feature_name = feature_type['name']

        log = feature_type['log']
        standardize = feature_type['standardize']
        feature_poly = feature_type['poly']
        print(
            f"Now transform for feature: {feature_name} by polynomial: {feature_poly}, natural log: {log}, standardize(z-score): {standardize}")

        X_train = X_trains[feature_name]
        X_test = X_tests[feature_name]

        if feature_poly:
            # make the features polynomial (order = 2)
            poly = PolynomialFeatures(2)
            X_train = poly.fit_transform(X_train)
            X_test = poly.fit_transform(X_test)

        if log:
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)

        if standardize:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            if np.ndim(X_test) == np.ndim(X_train):
                print(f"Test data standardized!")
                X_test = scaler.transform(X_test)

        X_trains[feature_name] = X_train
        X_tests[feature_name] = X_test

    return X_trains, X_tests, regulator_list_neighbor, regulator_list_self


def preprocess_concatenate_X_datasets(data_sets, meta_per_dataset, data_type, current_idxs,
                                      current_nns, feature_list, preprocess_X, cell_types_dict,
                                      neighbor_type='Del', log=False, per_cell=True,
                                      baseline=False, poly=True):
    """
    Integrate merFISH datasets for a given set of datasets. A wrapper function over the pre-
    -processing functions of X of each dataset. Concatenate preprocessed datasets and transform.
    For baseline covariates, add 1 more covariate: bregma.

    Args:
        data_sets: list of list of numpy arrays and dictionaries, in the format
            [current_meta, gene_exp, current_cor]
        log: boolean, if take transformation as log(x+1)
        preprocess_X: function, helper function to process different features for each cell/dataset(baseline)

    Return:
        X: numpy.array, concatenated accross datasets

    """

    # for general covariates
    X_all_list = []

    ## for each dataset, loop through each cell
    for i in range(len(data_sets)):
        current_meta, meta_columns, current_cor, cor_columns, \
        gene_exp, gene_exp_columns = data_sets[i]

        current_idx = current_idxs[i]

        if current_nns is not None:
            current_nn = current_nns[i]
        else:
            current_nn = None

        if meta_per_dataset is not None:
            meta_dataset = meta_per_dataset[i]
        else:
            meta_dataset = None

        if feature_list is not None:
            _all_features = list(gene_exp_columns.keys())
            feature_list_idx = [_all_features.index(f) for f in feature_list]

        if per_cell:
            x_list = []
            for i in range(0, len(current_idx)):
                x_conv = preprocess_X(i, data_type, current_meta, meta_columns, current_cor,
                                      cor_columns, gene_exp, gene_exp_columns, current_nn,
                                      current_idx, feature_list_idx, cell_types_dict, "Del")
                x_list.append(x_conv)

            _X = np.array(x_list)

            assert any(np.isnan(_X).flatten()) is False

            X_all_list.append(_X)

        else:
            _X = preprocess_X(data_type, current_meta, meta_columns, current_cor,
                              cor_columns, gene_exp, gene_exp_columns, current_nn,
                              current_idx, cell_types_dict, nn_type='Del', poly=poly)

            assert any(np.isnan(_X).flatten()) is False

            if baseline and data_type == 'merfish':
                # for baseline covariates: for each dataset, generate baseline covariates
                _, bregma = meta_dataset[0], float(meta_dataset[1])
                # treat bregma as continuous
                _feature_bregma = np.repeat(bregma, _X.shape[0])
                _X = np.concatenate([_feature_bregma[:, None], _X], axis=1)

            X_all_list.append(_X)

    # concatenate
    if len(X_all_list) > 0:
        X_all = np.concatenate(X_all_list, axis=0)
    else:
        X_all = np.array(X_all_list)

    del X_all_list

    # transformation; Not used for now
    #     if log:
    #         return np.log1p(X_all)
    #     else:
    return X_all
    
    
def preprocess_X_neighbor_by_type_per_cell(i, data_type, current_meta, meta_columns, current_cor, 
                                       cor_columns, gene_exp, gene_exp_columns, current_nn,
                                       current_idx, feature_list_idx, cell_types_dict,nn_type = 'Del'): 
    """
    Get neighboring expression as features for a single cell; sum the expression from different cell types separately.
    (not used currently)

    Args:
        i: integer, index of a cell in the current dataset
        data_type: string, merfish/starmap/merfish_cell_line
        current_meta: numpy array, meta info for the current dataset
        meta_columns: dictionary, use column name as key for column index
        current_cor: numpy array, spatial coordimates for the current dataset
        cor_columns: dictionary, use column name (e.g. cor_x, cor_y) as key for column index
        gene_exp: numpy array, gene expression for the current dataset
        gene_exp_columns: dictionary, use column name (gene names) as key for column index
        current_nn: numpy array, neighbors for cells in current dataset
        current_idx: list, index regarding the current dataset
        feature_list_idx: list of integers, index of the ligand to be used as neighboring features
        cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
        nn_type: string, type of neighbors, Del(Delaunay) or k-nn

    """

    idx_in_full = current_idx[i]
    _nn = current_nn[idx_in_full]
    
    if data_type == "seqfish":
        # seqfish's neighborhood defined only within FOV 
        fov = current_cor['Field of View'][idx_in_full]
        sub_gene_exp = gene_exp.loc[current_cor['Field of View'] == fov,:].copy()
        sub_gene_exp.reset_index(inplace=True, drop=True)
    else:
        # merfish's neighborhood defined over all cells 
        sub_gene_exp = gene_exp

    if nn_type == 'Del':
        nb_genes = sub_gene_exp[_nn,:]
    elif nn_type == 'k-nn':
        nb_genes = sub_gene_exp[_nn == 1]
    else:
        print ('Please specify neighborhood type')

    sub = current_meta[_nn, meta_columns['Cell_class']]

    nb_genes_fil = nb_genes[:, feature_list_idx]

    temp = cell_types_dict.copy()
    keys = temp.keys
    keys = list(temp.keys())

    _per_type = np.zeros([len(temp),len(feature_list_idx)])

    for cur_type in set(sub):
        _per_type[keys.index(cur_type),:] = np.sum(nb_genes_fil[np.where(sub == cur_type)[0],:], axis=0)

    _per_type = _per_type.reshape(-1)

    del nb_genes
    del sub
    del nb_genes_fil
        
    return _per_type



def preprocess_X_neighbor_per_cell(i, data_type, current_meta, meta_columns, current_cor, 
                                cor_columns, gene_exp, gene_exp_columns, current_nn,
                                current_idx, feature_list_idx, cell_types_dict,nn_type = 'Del'):
    """
     Get neighboring expression as features for a single cell; sum the expression from different cell jointly.

     Args:
         i: integer, index of a cell in the current dataset
         data_type: string, merfish/starmap/merfish_cell_line
         current_meta: numpy array, meta info for the current dataset
         meta_columns: dictionary, use column name as key for column index
         current_cor: numpy array, spatial coordimates for the current dataset
         cor_columns: dictionary, use column name (e.g. cor_x, cor_y) as key for column index
         gene_exp: numpy array, gene expression for the current dataset
         gene_exp_columns: dictionary, use column name (gene names) as key for column index
         current_nn: numpy array, neighbors for cells in current dataset
         current_idx: list, index regarding the current dataset
         feature_list_idx: list of integers, index of the ligand to be used as neighboring features
         cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
         nn_type: string, type of neighbors, Del(Delaunay) or k-nn

     """
        
    idx_in_full = current_idx[i]
    _nn = current_nn[idx_in_full]
    
    if data_type == "seqfish":
        # seqfish's neighborhood defined only within FOV 
        fov = current_cor['Field of View'][idx_in_full]
        sub_gene_exp = gene_exp.loc[current_cor['Field of View'] == fov,:].copy()
        sub_gene_exp.reset_index(inplace=True, drop=True)
    else:
        # merfish's neighborhood defined over all cells 
        sub_gene_exp = gene_exp

    if nn_type == 'Del':
        nb_genes = sub_gene_exp[_nn,:]
    elif nn_type == 'k-nn':
        nb_genes = sub_gene_exp[_nn == 1]
    else:
        print ('Please specify neighborhood type')

    nb_genes_fil = nb_genes[:, feature_list_idx]
    
    # choose a convlution method 
    # x_conv = np.log2(nb_genes_fil + 1).sum()
    x_conv = np.sum(nb_genes_fil, axis = 0)
    
    del nb_genes_fil
        
    return x_conv 

def preprocess_X_self_per_cell(i, data_type, current_meta, meta_columns, current_cor, 
                                cor_columns, gene_exp, gene_exp_columns, current_nn,
                                current_idx, feature_list_idx, cell_types_dict,nn_type = 'Del'):
    """
     Get cell's own expression as features for a single cell.
     Args:
         i: integer, index of a cell in the current dataset
         data_type: string, merfish/starmap/merfish_cell_line
         current_meta: numpy array, meta info for the current dataset
         meta_columns: dictionary, use column name as key for column index
         current_cor: numpy array, spatial coordimates for the current dataset
         cor_columns: dictionary, use column name (e.g. cor_x, cor_y) as key for column index
         gene_exp: numpy array, gene expression for the current dataset
         gene_exp_columns: dictionary, use column name (gene names) as key for column index
         current_nn: numpy array, neighbors for cells in current dataset
         current_idx: list, index regarding the current dataset
         feature_list_idx: list of integers, index of the ligand to be used as neighboring features
         cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
         nn_type: string, type of neighbors, Del(Delaunay) or k-nn

     """

    idx_in_full = current_idx[i]
    x_self = gene_exp[idx_in_full, feature_list_idx]
        
    return x_self




def preprocess_X_neighbor_type_per_dataset(data_type, current_meta, meta_columns, current_cor, 
                                       cor_columns, gene_exp, gene_exp_columns, current_nn,
                                       current_idx, cell_types_dict, nn_type = 'Del', poly = True):
    """
     Get cell's neighboring cell types for a single cell.

     Args:
         i: integer, index of a cell in the current dataset
         data_type: string, merfish/starmap/merfish_cell_line
         current_meta: numpy array, meta info for the current dataset
         meta_columns: dictionary, use column name as key for column index
         current_cor: numpy array, spatial coordimates for the current dataset
         cor_columns: dictionary, use column name (e.g. cor_x, cor_y) as key for column index
         gene_exp: numpy array, gene expression for the current dataset
         gene_exp_columns: dictionary, use column name (gene names) as key for column index
         current_nn: numpy array, neighbors for cells in current dataset
         current_idx: list, index regarding the current dataset
         feature_list_idx: list of integers, index of the ligand to be used as neighboring features
         cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
         nn_type: string, type of neighbors, Del(Delaunay) or k-nn

     """

    x_list = []
    for i in range(0,len(current_idx)):
        idx_in_full = current_idx[i]
        _nn = current_nn[idx_in_full]
        sub= current_meta[_nn,:]
        # not count in cell with 'NaN' in their cell class 
        cell_types_raw = sub[:, meta_columns['Cell_class']]
        cell_types_valid = cell_types_raw[~pd.isnull(cell_types_raw)]
        unique, counts = np.unique(cell_types_valid, return_counts=True)
        x_conv = dict(zip(unique, counts))
        temp = cell_types_dict.copy()
        for key, value in temp.items():
            if key in x_conv:
                temp[key] = x_conv[key]
        x_list.append(list(temp.values()))

    X_n_type = np.array(x_list)

    del x_list
    
    return X_n_type

def preprocess_X_baseline_per_dataset(data_type, current_meta, meta_columns, current_cor, 
                                       cor_columns, gene_exp, gene_exp_columns, current_nn,
                                       current_idx, cell_types_dict, nn_type = 'Del', poly = False):
    """
     Get cell's baselines (spatial coordinates) as features for a single cell.

     Args:
         i: integer, index of a cell in the current dataset
         data_type: string, merfish/starmap/merfish_cell_line
         current_meta: numpy array, meta info for the current dataset
         meta_columns: dictionary, use column name as key for column index
         current_cor: numpy array, spatial coordimates for the current dataset
         cor_columns: dictionary, use column name (e.g. cor_x, cor_y) as key for column index
         gene_exp: numpy array, gene expression for the current dataset
         gene_exp_columns: dictionary, use column name (gene names) as key for column index
         current_nn: numpy array, neighbors for cells in current dataset
         current_idx: list, index regarding the current dataset
         feature_list_idx: list of integers, index of the ligand to be used as neighboring features
         cell_types_dict: dictionary, cell type as key and 0 as value; include all available cell types
         nn_type: string, type of neighbors, Del(Delaunay) or k-nn

     """
   

    # prepare baseline features 
    if data_type == 'seqfish':
        # for the categories not present, fill them all 0
        dummies = pd.get_dummies(current_meta['Cell_class'][current_idx], prefix='', prefix_sep='')
        full_cell_types = list(cell_types_dict.keys())
        cell_type_one_hot  = dummies.T.reindex(full_cell_types).T.fillna(0)
        cell_type_one_hot.reset_index(inplace=True, drop=True)
        cor_sub = current_cor[['X','Y']].iloc[current_idx,:]
        cor_sub.reset_index(inplace=True, drop=True)
        X = pd.concat([cor_sub,cell_type_one_hot],axis=1)
    elif 'merfish' in data_type or data_type == 'starmap':
        if current_meta is not None: 
            full_cell_types = list(cell_types_dict.keys())
            # for the categories not present, fill them all 0
            dummies = pd.get_dummies(current_meta[current_idx,meta_columns['Cell_class']], prefix='', prefix_sep='')
            cell_type_one_hot = dummies.T.reindex(full_cell_types).T.fillna(0)
            cell_type_one_hot.reset_index(inplace=True, drop=True)
            cell_type_one_hot = cell_type_one_hot .to_numpy()
        if current_cor is not None:
            cor_sub = current_cor[current_idx,:]
            
        if current_meta is not None and current_cor is not None:
            X = np.concatenate([cor_sub, cell_type_one_hot],axis=1)
            del cell_type_one_hot
            del dummies
            del cor_sub
    
        elif current_meta is not None:
            X = cell_type_one_hot
        elif current_cor is not None:
             X = cor_sub
        else:
            raise TypeError("Both cell type info and coordinates info NOT available! No baselie featues "
                            "should be generated")
            return None
    else:
        raise NotImplementedError (f"Only support merfish/starmap/mefish_cell_line")

        return None
    
    if poly:
        # make the features polynomial (order = 2)
        poly = PolynomialFeatures(2)
        X = poly.fit_transform(X)
        
    
    return X


# --- process Y 


def prepare_responses(data_type, datasets_train, datasets_test, idx_train_in_general, idx_test_in_general,
                      idx_train_in_dataset, idx_test_in_dataset, neighbors_train, neighbors_test,
                      response_type, log_response, response_list_prior, top_k_response,
                      genes_filtered, l_u, r_u):
    """

    Generate response variables.

    Args:
        data_type: string, merfish/starmap/merfish_cell_line
        datasets_train: list of numpy arrays & dictionaries; all info for training samples;in the format
           [current_meta, gene_exp, current_cor]
        datasets_test: list of numpy arrays & dictionaries; all info for testing samples;in the format
           [current_meta, gene_exp, current_cor]
        idx_train_in_general:list, index of training cells in the general (all cell types) list
        idx_test_in_general:list, index of testing cells in the general (all cell types) list
        idx_train_in_dataset: list of list of index of training cells in each dataset (as grouped by animal & bregma)
        idx_test_in_dataset: list of list of index of testing cells in each dataset (as grouped by animal & bregma)
        neighbors_train: list of numpy arrays, neighboring info for the training samples
        neighbors_test: list of numpy arrays, neighboring info for the testing samples
        response_type: string, if use raw expression value as response ('original')
        log_response: boolean, if use log transformed expression values for response
        response_list_prior: list of strings, list of responses genes from prior
        top_k_response: integer, top K response genes to be used as response; they will be sorted \
        according to dispersion (descending); default None
        genes_filtered: list of strings, all genes profiled in the expression data; all letters are in uppercase
        l_u: list of strings, ligands from the database; all letters are in uppercase
        r_u: list of strings, receptors from the database; all letters are in uppercase

       Returns:
           Y_train: numpy array, sample size x number of responses, processed values (can still be same as _true if \
           response_type = 'original')
           Y_train_true: numpy array, sample size x number of responses, raw values
           Y_test: numpy array, sample size x number of responses, processed values (can still be same as _true if \
           response_type = 'original')
           Y_test_true: numpy array, sample size x number of responses, raw values
           response_list: list of strings, names of genes used for responses

       """

    if response_list_prior is not None:
        feature_list = response_list_prior
        response_list = feature_list
    else:
        feature_list = [g for g in genes_filtered if g not in l_u.union(r_u)]
        # feature_list = genes_filtered
        response_list = feature_list

    response_type = response_type
    log_response = log_response
    print(f"Response type: {response_type}; transformby natural log: {log_response}")
    bins = [-1e10, -0.3, 0.3, 1e10]

    data_sets = datasets_train
    current_idxs = idx_train_in_dataset
    current_nns = neighbors_train
    current_idx_in_general = idx_train_in_general
    baseline_file = None
    # baseline_file = f"output/baseline/{data_type}/general/{preprocess}/{test_animal}/Y_pred_train_{behavior}_{sex}.npy"

    Y_train, Y_train_true, transformer = preprocess_Y_datasets(data_sets, data_type, current_idxs,
                                                               current_idx_in_general,
                                                               feature_list, bins, response_type,
                                                               baseline_file=baseline_file, log=False, transformer=None,
                                                               scale=False)

    data_sets = datasets_test
    current_idxs = idx_test_in_dataset
    current_nns = neighbors_test
    current_idx_in_general = idx_test_in_general
    baseline_file = None
    # baseline_file = f"output/baseline/{data_type}/general/{preprocess}/{test_animal}/Y_pred_test_{behavior}_{sex}.npy"

    Y_test, Y_test_true, _ = preprocess_Y_datasets(data_sets, data_type, current_idxs, current_idx_in_general,
                                                   feature_list, bins, response_type, baseline_file=baseline_file,
                                                   log=False, transformer=transformer, scale=False)

    # filter response genes
    if top_k_response is not None:
        dispersion = Y_train.var(axis=0) / (Y_train.mean(axis=0) + 1e-6)

        idx_filtered = np.argsort(-dispersion, axis=0)[:top_k_response]
        Y_train = Y_train[:, idx_filtered]
        Y_train_true = Y_train_true[:, idx_filtered]
        if np.ndim(Y_test) > 1:
            Y_test = Y_test[:, idx_filtered]
            Y_test_true = Y_test_true[:, idx_filtered]
        response_list = list(np.array(response_list)[idx_filtered])

    if log_response:
        Y_train = np.log1p(Y_train)
        Y_test = np.log1p(Y_test)

    return Y_train, Y_train_true, Y_test, Y_test_true, response_list


def preprocess_Y_datasets(data_sets, data_type, current_idxs, current_idx_in_general,
                          feature_list, bins, response_type, baseline_file=None, log=False, transformer=None,
                          scale=False):
    """

    Preprocess for response variables Y for a data sets. The dataset contains info for a single animal's bregma.

    Args:
        data_sets: list of list of numpy arrays and dictionaries, in the format [current_meta, gene_exp, current_cor]
        response_type: string, one of ['original', 'residual', 'residual_category']
        current_idxs: list of list of index of cells in each dataset
        current_idx_in_general: list, sub-group index in all datasets(all animals/FOVs)
    Return:
        Y_processed: numpy array, processed response variables
        Y_true: numpy array, original values
        transformer: None or a StandardScaler object

    """

    # Prepare the true/original Y
    Y_true_all = []

    for i in range(len(data_sets)):
        current_meta, meta_columns, current_cor, cor_columns, \
        gene_exp, gene_exp_columns = data_sets[i]
        current_idx = current_idxs[i]

        # Prepare the true: subset the true value
        Y_true = gene_exp[current_idx, :].copy()
        Y_true_all.append(Y_true)

    # concatenate the original Y for all datasets
    if len(Y_true_all) == 1:
        Y_true = Y_true_all[0]
    elif len(Y_true_all) == 0:
        Y_true = np.array([])
    else:
        Y_true = np.concatenate(Y_true_all, axis=0)

    # Prepare the true: take the log
    if log:
        Y_true = np.log1p(Y_true)

        # Prepare the true: scale the true
    if scale:
        # scaling is done for ALL datasets together!
        if transformer is None:
            # Given that the baseline scaler is fit over entire train data,
            # should apply to the entire train data here as well
            transformer = preprocessing.StandardScaler().fit(gene_exp)
            Y_true = transformer.transform(Y_true)
            print('Scaled using new transformer on gene_exp!')
        else:
            Y_true = transformer.transform(Y_true)
            print('Scaled using old transformer!')

            # get subset for feature list
    if feature_list is not None and len(Y_true_all) > 0:
        _all_features = list(gene_exp_columns.keys())
        feature_list_idx = [_all_features.index(f) for f in feature_list]

    if response_type == 'original':
        if len(Y_true_all) > 0:
            return Y_true[:, feature_list_idx], Y_true[:, feature_list_idx], transformer
        else:
            return Y_true, Y_true, transformer

    elif baseline_file is not None:

        Y_pred_baseline = np.load(f"{baseline_file}")

        # subset the prediction
        if current_idx_in_general is not None:
            Y_pred_baseline_sub = Y_pred_baseline.T[current_idx_in_general, :]
        else:
            print('Current_idx_in_general is None!')

        #         if log: # already taken log before fitting; no need
        #             Y_pred_baseline_sub = np.log1p(Y_pred_baseline_sub)

        Y_dif = Y_true[:, feature_list_idx] - Y_pred_baseline_sub

        if response_type == 'residual':
            return Y_dif, Y_true[:, feature_list_idx], transformer
        else:  # TBD
            temp = Y_dif[feature_list].apply(lambda x: pd.cut(x, bins, labels=[0, 1, 2]), axis=1)
            return temp, transformer
    else:

        print("Please provide the filename of the baseline prediction.")
        return '_', '_', '_'


# --- filter

flatten = lambda l: [item for sublist in l for item in sublist]

def filter_by_response(Y_processed, Y_true, criterion = 'non-zero'):
    """

    Args:
        Y_processed: numpy array, processed response variables
        Y_true: numpy rray, original values

    Returns:
        idx_per_response: list of list of indexes of each response variable; order as in Y_true

    """
    
    idx_per_response = []
    for i in range(0,Y_true.shape[1]):
        
        if criterion == 'non-zero':
            _idx = np.where(Y_true[:,i] != 0)[0]
        
        idx_per_response.append(_idx)
    
    return idx_per_response

# --- combine features
def combine_features(features, preprocess, num_coordinates):
    """
    Features: dictionary of numpy array; presumbly each having dimension >1

    """
    if 'neighbor' not in preprocess:

        X = np.concatenate((features['regulators_neighbor_self'], features['regulators_self']), axis=1)
        X_clf = X
        X_full = X

    else:

        if 'neighbor_cat' in preprocess:
            X = np.concatenate(
                (features['regulators_neighbor_self'], features['regulators_self'], features['regulators_neighbor']),
                axis=1)

        elif 'neighbor_sum' in preprocess:
            X = np.concatenate(
                (features['regulators_neighbor_self'] + features['regulators_neighbor'], features['regulators_self']),
                axis=1)

        if features['neighbor_type'] is not None and features['baseline'] is not None:
            X_clf = np.concatenate([features['baseline'][:, :num_coordinates], features['neighbor_type']], axis=1)
            X_full = np.concatenate([X, X_clf], axis=1)

        elif features['neighbor_type'] is not None:
            X_clf = np.concatenate([features['neighbor_type']], axis=1)
            X_full = np.concatenate([X, X_clf], axis=1)


        elif features['baseline'] is not None:
            X_clf = np.concatenate([features['baseline'][:, :num_coordinates]], axis=1)
            X_full = np.concatenate([X, X_clf], axis=1)


        else:
            X_clf = X
            X_full = X


    return X, X_clf, X_full
