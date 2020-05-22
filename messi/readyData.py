"""
Download and preprocess a helper dataset and an expression dataset.

The script will first download the ligand and receptor list.

Then, for a particular type of expression data, the script first downloads it at ta user-defined folder. It then preprocess
it (by cutting the whole dataset into subsets based on animal/sample ID and other info (e.g., bregma)) and
produces files with meta informations. All preprocessed files will be saved at the same user-defined folder.

This file can also be imported as a module and contains the following
functions:
    * process_merfish_raw - save samples separately from the downloaded file; also save meta info files
    * main - the main function of the script

"""



import os
import argparse
import urllib.request
import shutil

import numpy as np
import pandas as pd

from messi.data_processing import *



def process_merfish_raw(input_path, raw_filename= 'merfish_raw.csv'):

    merfish = pd.read_csv(input_path + raw_filename, header=0)
    print(f"The raw sample shape is {merfish.shape}")

    sex = 'Female'
    behaviors = ['Naive', 'Parenting', 'Virgin Parenting']

    for behav in behaviors:
        behav_no_space = behav.replace(" ", "_")
        print(f"Process data for {behav} of {sex} animals")

        all_behavior = merfish[merfish['Behavior'] == behav]
        animals = all_behavior['Animal_ID'].value_counts().index.values
        bregmas = all_behavior['Bregma'].value_counts().index.values

        # save cells separately according to the animal ID and bregma
        print (f"Save data separately for different animals and bregmas.")
        for a in animals:
            print(f"Animal {a}")
            for b in bregmas:
                sample = all_behavior[all_behavior['Animal_ID'] == a]
                sample = sample[sample['Bregma'] == b]
                sample.reset_index(inplace=True, drop=True)
                if sample.shape[0] != 0:
                    b_format = ''.join(str(b).split('.'))
                    print(f"Formatted bregma {b_format}")
                    filename = f"merfish_animal{a}_bregma{b_format}.csv"
                    print(f"Saving to {os.path.join(input_path, filename)}")
                    sample.to_csv(os.path.join(input_path, filename), sep=',', index=False)
                else:
                    print(f"No data meeting requirements: {sample.shape}")

        # save files of meta information
        print (f"Save meta information for different behaviors and genders.")
        meta_columns = ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Cell_class']
        subset = merfish[meta_columns][(merfish['Behavior'] == behav) & (merfish['Animal_sex'] == sex)]

        subset.reset_index(inplace=True, drop=True)
        subset['ID_in_dataset'] = np.repeat(-1, subset.shape[0])

        animals = subset['Animal_ID'].value_counts().index.values
        bregmas = subset['Bregma'].value_counts().index.values

        uniques = []
        for a in animals:
            for b in bregmas:
                condition = (subset['Animal_ID'] == a) & (subset['Bregma'] == b)
                sample = subset[condition]

                # check if index have appeared before
                index_new = sample.index.values
                assert len(set(index_new).intersection(set(uniques))) == 0
                uniques = uniques + list(index_new)

                if sample.shape[0] != 0:
                    # b_format = ''.join(str(b).split('.'))
                    # print(b_format)
                    subset.loc[condition, 'ID_in_dataset'] = list(range(0, sample.shape[0]))
                else:
                    print(sample.shape)

        assert any(subset['ID_in_dataset'] == -1) is False

        filename = f"merfish_meta_{behav_no_space}_{sex}.csv"
        print(f"Saving to {os.path.join(input_path, filename)}")
        subset.to_csv(os.path.join(input_path, filename), sep=',', index=False)



def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        default='input/', help="string, path to the folder to save the expression data, "
                                                  "default 'input/'")
    parser.add_argument('-d', '--dataType', required=True,
                        default='merfish', help="string, type of expression data, default 'merfish'")


    args = parser.parse_args()

    # set parameters for data
    input_path = args.input
    data_type = args.dataType

    # create folder to save data
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # set parameters for downloading the helper data
    url = "https://github.com/doraadong/MESSI/blob/master/messi/input/ligand_receptor_pairs2.txt"

    # download data
    filename = "ligand_receptor_pairs2.txt"
    if os.path.exists(os.path.join(input_path, filename)):
        print(f"{os.path.join(input_path, filename)} alrady exists. No need downloading.")
    else:
        print(f"Downloading ligand/receptor list to {os.path.join(input_path, filename)}")
        with urllib.request.urlopen(url) as response, open(os.path.join(input_path, filename), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    # set parameters specific to the data type
    if data_type == 'merfish':
        url = "https://datadryad.org/stash/downloads/file_stream/68364"
        process_raw = process_merfish_raw
    else:
        raise NotImplementedError(f"Now only support processing MERFISH hypothalamus data.")

    # download data
    filename = f"{data_type}_raw.csv"
    if os.path.exists(os.path.join(input_path, filename)):
        print(f"{os.path.join(input_path, filename)} alrady exists. No need downloading.")
    else:
        print(f"Downloading expression data to {os.path.join(input_path, filename)}")
        with urllib.request.urlopen(url) as response, open(os.path.join(input_path, filename), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    # process data
    process_raw(input_path)


if __name__ == '__main__':
    main()
