import pandas as pd
import numpy as np
import logging
import os
import random
import json
import sys
import glob
import gc
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from src.utils import split_masked_cells


class GZSL_data_loader(object):
    def __init__(self, dataset, remove_col, device='cuda'):
        """
        Data loader class for Generalized Zero-Shot Learning (GZSL).

        This class is responsible for loading and preprocessing the GZSL dataset,
        dividing it into source and target domains, and providing methods to retrieve
        batches of data for training.

        Args:
            dataset (dict): A dictionary containing the dataset information.
                            The dictionary should have the following keys:
                                - 'X_source': Source domain features.
                                - 'y_source': Source domain labels.
                                - 'X_train': Target domain training features.
                                - 'y_train': Target domain training labels.
                                - 'X_test': Target domain test features.
                                - 'y_test': Target domain test labels.
            remove_col (int): Number of masked cells to remove.
            device (str, optional): Device to use for computation (default: 'cuda').

        """
        self.X_source = dataset['X_source']
        self.y_source = dataset['y_source']
        self.X_train = dataset['X_train']
        self.y_train = dataset['y_train']
        self.X_test = dataset['X_test']
        self.y_test = dataset['y_test']
        self.remove_col = remove_col

        # Split the masked cells in the training data into seen and unseen classes
        self.X_seen, self.X_unseen, self.y_seen, self.y_unseen = split_masked_cells(self.X_train, self.y_train, masked_cells=remove_col)
        self.ntrain = len(self.y_seen)
        self.ntest = len(self.y_test)

    def next_batch(self, batch_size):
        """
        Get the next batch of data for training.

        Args:
            batch_size (int): Size of the batch to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - target: Target domain batch features.
                - source: Source domain batch features.
                - target_label: Target domain batch labels.
                - source_label: Source domain batch labels.

        """
        target, source = [], []
        target_label, source_label = [], []
        for i in list(set(self.y_seen)):
            target_index = np.in1d(self.y_seen, i, invert=False)
            source_index = np.in1d(self.y_source, i, invert=False)
            target_subset = self.X_seen[target_index, :]
            source_subset = self.X_source[source_index, :]
            target_label_subset = self.y_seen[target_index]
            source_label_subset = self.y_source[source_index]

            target_index = list(range(len(target_subset)))
            source_index = list(range(len(source_subset)))
            random.shuffle(target_index)
            random.shuffle(source_index)

            target_subset = target_subset[target_index[:5], :]
            source_subset = source_subset[source_index[:5], :]
            target.append(target_subset)
            source.append(source_subset)
            target_label.append(target_label_subset[target_index[:5]])
            source_label.append(source_label_subset[source_index[:5]])

        target, source, target_label, source_label = map(np.concatenate, [target, source, target_label, source_label])
        return target, source, target_label, source_label


def preset_mousehuman_load(root_dir, organ_name, prep):
    """
    Load a preset Mouse-Human dataset.

    This function loads a Mouse-Human dataset based on the specified organ and
    preprocessing method.

    Args:
        root_dir (str): Root directory of the dataset.
        organ_name (str): Name of the organ.
        prep (str): Preprocessing method.

    Returns:
        dict: A dictionary containing the dataset information.
              The dictionary has the following keys:
                  - 'X_source': Source domain features.
                  - 'y_source': Source domain labels.
                  - 'X_train': Target domain training features.
                  - 'y_train': Target domain training labels.
                  - 'X_test': Target domain test features.
                  - 'y_test': Target domain test labels.

    """
    organ_dic = {
        'bm': "bm/bm_",
        'brain': "brain/brain_",
        'pancreas': "pancreas/pancreas_",
    }
    prep_dic = {
        'scanpy': "scanpy_pca.csv",
        'dca': "dca_pca.csv",
        'scetm': "scetm.csv"
    }

    organ_dir = organ_dic[organ_name]
    prep_dir = prep_dic[prep]
    prep_name = prep_dir.rstrip('.csv')

    PATH_in_source = root_dir + organ_dir + "mouse_red_" + prep_dir
    PATH_in_source_label = root_dir + organ_dir + "mouse_red_label.csv"
    PATH_in_target_train = root_dir + organ_dir + "human_red_train_" + prep_dir
    PATH_in_target_train_label = root_dir + organ_dir + "human_red_train_label.csv"
    PATH_in_target_test = root_dir + organ_dir + "human_red_test_" + prep_dir
    PATH_in_target_test_label = root_dir + organ_dir + "human_red_test_label.csv"
    
    X_source = pd.read_csv(PATH_in_source, index_col=0)
    X_target = pd.read_csv(PATH_in_target_train, index_col=0)
    target_columns = X_target.columns
    source_columns = X_source.columns

    dataset = {
        'X_source': X_source.to_numpy(),
        'y_source': pd.read_csv(PATH_in_source_label, index_col=0)["label"].to_numpy("int32"),
        'X_train': X_target.to_numpy(),
        'y_train': pd.read_csv(PATH_in_target_train_label, index_col=0)["label"].to_numpy("int32"),
        'X_test': pd.read_csv(PATH_in_target_test, index_col=0).to_numpy(),
        'y_test': pd.read_csv(PATH_in_target_test_label, index_col=0)["label"].to_numpy("int32"),
        'id_source' : source_columns,
        'id_target' : target_columns,
        'fi_source' : None,
        'fi_target' : None,
    }

    return dataset


def _get_dataframe(flist):
    """
    Helper function to concatenate dataframes.

    Args:
        flist (list): List of file paths.

    Returns:
        pd.DataFrame: Concatenated dataframe.

    """
    tb = pd.DataFrame([])
    for f in flist:
        v = pd.read_csv(f, sep=',', header=0, index_col=0)
        tb = pd.concat([tb, v], axis=1)
    tb = tb.transpose()
    return tb


def _get_data(root_dir: str, mo: str):
    """
    Helper function to load data from files.

    Args:
        root_dir (str): Root directory of the dataset.
        mo (str): MO value.

    Returns:
        tuple: A tuple containing the following elements:
            - dataset: Loaded dataset.
            - labels: Labels associated with the dataset.

    """
    dataset_unst_files = glob.glob(f'{root_dir}{mo}*unst*.csv')
    dataset_lps2_files = glob.glob(f'{root_dir}{mo}*lps2*.csv')
    dataset_lps4_files = glob.glob(f'{root_dir}{mo}*lps4*.csv')
    dataset_lps6_files = glob.glob(f'{root_dir}{mo}*lps6*.csv')
    s_unst = _get_dataframe(dataset_unst_files)
    s_lps2 = _get_dataframe(dataset_lps2_files)
    s_lps4 = _get_dataframe(dataset_lps4_files)
    s_lps6 = _get_dataframe(dataset_lps6_files)

    dataset = pd.concat([s_unst, s_lps2, s_lps4, s_lps6], axis=0)
    labels = [0] * len(s_unst) + [1] * len(s_lps2) + [2] * len(s_lps4) + [3] * len(s_lps6)
    del(s_unst, s_lps2, s_lps4, s_lps6)
    gc.collect()
    return dataset, labels


def lps_stimulate_load(root_dir: str, source: str, target: str, latent_dim=None):
    """
    Load the LPS stimulation dataset.

    This function loads the LPS stimulation dataset based on the specified
    source and target parameters.

    Args:
        root_dir (str): Root directory of the dataset.
        source (str): Source dataset.
        target (str): Target dataset.
        latent_dim (object, optional): Latent dimension object for transformation (default: None).

    Returns:
        dict: A dictionary containing the dataset information.
              The dictionary has the following keys:
                  - 'X_source': Source domain features.
                  - 'y_source': Source domain labels.
                  - 'X_train': Target domain training features.
                  - 'y_train': Target domain training labels.
                  - 'X_test': Target domain test features.
                  - 'y_test': Target domain test labels.

    """
    target, target_labels = _get_data(root_dir, target)
    source, source_labels = _get_data(root_dir, source)
    target_labels = np.array(target_labels, dtype='int32')
    source_labels = np.array(source_labels, dtype='int32')
    target_columns = target.columns
    source_columns = source.columns
    if latent_dim is not None:
        target = latent_dim.fit_transform(target)
        target_fi = latent_dim.components_
        source = latent_dim.fit_transform(source)
        source_fi = latent_dim.components_
    else:
        target = np.array(target)
        source = np.array(source)
        target_fi = None
        source_fi = None

    SKF = StratifiedKFold(n_splits=5).split(target, target_labels)
    for train_index, test_index in SKF:
        hX_train, hX_test = target[train_index, :], target[test_index, :]
        hy_train, hy_test = target_labels[train_index], target_labels[test_index]

        dataset = {
            'X_source': source,
            'y_source': source_labels,
            'X_train': hX_train,
            'y_train': hy_train,
            'X_test': hX_test,
            'y_test': hy_test,
            'id_source' : source_columns,
            'id_target' : target_columns,
            'fi_source' : source_fi,
            'fi_target' : target_fi,
        }

        return dataset


def baron_load(root_dir, latent_dim=None):
    """
    Load the Baron dataset.

    This function loads the Baron dataset for cross-species analysis.

    Args:
        latent_dim (object, optional): Latent dimension object for transformation (default: None).

    Returns:
        tuple: A tuple containing the following elements:
            - dataset: Loaded dataset.
            - common_set: Set of common labels between source and target domains.

    """
    target = pd.read_csv(f'{root_dir}norm_human.csv', index_col=0, header=0)
    target_labels = pd.read_csv(f'{root_dir}norm_human_label.csv', index_col=0, header=0)
    source = pd.read_csv(f'{root_dir}norm_mouse.csv', index_col=0, header=0)
    source_labels = pd.read_csv(f'{root_dir}norm_mouse_label.csv', index_col=0, header=0)
    target = target.T
    source = source.T
    target_columns = target.columns
    source_columns = source.columns
    target_labels = target_labels.T.values.tolist()[0]
    source_labels = source_labels.T.values.tolist()[0]
    ##
    common_set = set(target_labels) & set(source_labels)
    all_set = set(target_labels) | set(source_labels)
    out_set = all_set - common_set
    le = preprocessing.LabelEncoder()
    le.fit(list(common_set))

    target_labels = np.array(target_labels)
    source_labels = np.array(source_labels)
    keep = np.in1d(target_labels, list(common_set), invert=False)
    target = target.loc[keep, :]
    target_labels = target_labels[keep]
    del(keep)
    keep = np.in1d(source_labels, list(common_set), invert=False)
    source = source.loc[keep, :]
    source_labels = source_labels[keep]
    del(keep)

    common_set = le.transform(list(common_set))
    ###

    target_labels = np.array(le.transform(target_labels), dtype='int32')
    source_labels = np.array(le.transform(source_labels), dtype='int32')

    if latent_dim is not None:
        target = latent_dim.fit_transform(target)
        target_fi = latent_dim.components_
        source = latent_dim.fit_transform(source)
        source_fi = latent_dim.components_
    else:
        target = np.array(target)
        source = np.array(source)
        target_fi = None
        source_fi = None


    SKF = StratifiedKFold(n_splits=5).split(target, target_labels)
    for train_index, test_index in SKF:
        hX_train, hX_test = target[train_index, :], target[test_index, :]
        hy_train, hy_test = target_labels[train_index], target_labels[test_index]

        dataset = {
            'X_source': source,
            'y_source': source_labels,
            'X_train': hX_train,
            'y_train': hy_train,
            'X_test': hX_test,
            'y_test': hy_test,
            'id_source' : source_columns,
            'id_target' : target_columns,
            'fi_source' : source_fi,
            'fi_target' : target_fi,
        }

        return dataset, common_set
