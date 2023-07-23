import pandas as pd
import numpy as np
import anndata as ad
import logging
import os
import json
import sys
import h5py
import multiprocessing
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
from itertools import combinations
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon


def create_logger(jobid, current_time):
    """
    Creates a logger for logging messages during program execution.

    Args:
    	jobid (str): The identifier for the current job.
    	current_time (str): The timestamp for the current time.

    Returns:
    	logger (Logger): The logger object for logging messages.
    """
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s %(message)s'
    )

    formatter = logging.Formatter(\
        '[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    handler = logging.FileHandler(f'./job_{jobid}_time_{current_time}.log')
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers): 
        logger.addHandler(handler)
    return logger


def h_score(y_true, y_pred, masked_cells):
    """
    Calculates the H score, accuracy of known classes, and accuracy of unknown classes.

    Args:
    	y_true (numpy.ndarray): The true labels.
    	y_pred (numpy.ndarray): The predicted labels.
    	masked_cells (list): A list of masked cells.

    Returns:
    	h (float): The H score.
    	acc_known (float): The accuracy of known classes.
    	acc_unknown (float): The accuracy of unknown classes.
    """
    warnings.filterwarnings('ignore')
    known = np.in1d(y_true, masked_cells, invert=True)
    acc_known = balanced_accuracy_score(y_true[known], y_pred[known])
    acc_unknown = balanced_accuracy_score(y_true[~known], y_pred[~known])
    h = (2 * acc_known * acc_unknown) / (acc_known + acc_unknown)
    return h, acc_known, acc_unknown


def split_masked_cells(X_t, y_t, masked_cells, balance=False, n=500):
    """
    Masks cells for generalized zero-shot learning.

    Args:
	X_t (numpy.ndarray): The feature matrix of the target data.
	y_t (numpy.ndarray): The labels of the target data.
	masked_cells (list): A list of cells to be masked from the data.
	balance (bool): Whether to balance seen train data.
	n (int): The desired number of samples per class.

    Returns:
	X_t_seen (numpy.ndarray): The features of seen classes.
	X_t_unseen (numpy.ndarray): The features of unseen classes.
	y_seen (numpy.ndarray): The labels of seen classes.
	y_unseen (numpy.ndarray): The labels of unseen classes.
    """
    keep = np.in1d(y_t, masked_cells, invert=True)
    X_t_seen = X_t[keep]
    X_t_unseen = X_t[~keep]
    y_seen = y_t[keep]
    y_unseen = y_t[~keep]
    if balance:
        X_t_seen, y_seen = balance_sampling(X_t_seen, y_seen, n)
    return X_t_seen, X_t_unseen, y_seen, y_unseen


def balance_sampling(X, y, n=100):
    """
    Re-balances data by over-sampling with SMOTE and under-sampling randomly.

    Parameters:
	X (numpy.ndarray): The feature matrix.
	y (numpy.ndarray): The labels.
	n (int): The desired samples per class.

    Returns:
	X_combined_sampling (numpy.ndarray): The resampled feature matrix.
	y_combined_sampling (numpy.ndarray): The resampled labels.
    """
    warnings.filterwarnings('ignore')
    counts = Counter(y)
    under = np.array([], dtype="int32")
    over = np.array([], dtype="int32")
    for i in counts.keys():
        if counts[i] <= n:
            over = np.concatenate((over, np.array([i])))
        else:
            under = np.concatenate((under, np.array([i])))
    if len(over) == 0:
        dict_under = dict(zip(under, [n for i in range(len(under))]))
        under_sam =  RandomUnderSampler(sampling_strategy=dict_under)
        X_under, y_under = under_sam.fit_resample(X, y)
        return X_under, y_under
    elif len(under) == 0:
        dict_over = dict(zip(over, [n for i in range(len(over))]))
        over_sam = SMOTE(sampling_strategy=dict_over)
        X_over, y_over = over_sam.fit_resample(X, y)
        return X_over, y_over
    else:
        if len(over) == 1:
            # Tricks SMOTE into oversampling one class
            pseudo_X = np.full((n, X.shape[1]), 10000)
            pseudo_y = np.full(n, 10000)
            dict_over = dict()
            dict_over[over[0]] = n
            dict_over[10000] = n
            is_over = np.in1d(y, over)
            over_sam = SMOTE(sampling_strategy=dict_over)
            is_over = np.in1d(y, over)
            X_over_, y_over_ = over_sam.fit_resample(np.concatenate((X[is_over], pseudo_X)),
                                                     np.concatenate((y[is_over], pseudo_y)))
            X_over = X_over_[y_over_==over[0]]
            y_over = y_over_[y_over_==over[0]]

        else:
            dict_over = dict(zip(over, [n for i in range(len(over))]))
            over_sam = SMOTE(sampling_strategy=dict_over)
            is_over = np.in1d(y, over)
            X_over, y_over = over_sam.fit_resample(X[is_over], y[is_over])

        if len(under) == 1:
            # Tricks RandomUnderSampler into working with one class
            pseudo_X = np.full((n, X.shape[1]), 10000)
            pseudo_y = np.full(n, 10000)
            dict_under = dict()
            dict_under[under[0]] = n
            dict_under[10000] = n
            is_under = np.in1d(y, under)
            under_sam = RandomUnderSampler(sampling_strategy=dict_under)
            is_under = np.in1d(y, under)
            X_under_, y_under_ = under_sam.fit_resample(np.concatenate((X[is_under], pseudo_X)),
                                                        np.concatenate((y[is_under], pseudo_y)))
            X_under = X_under_[y_under_==under[0]]
            y_under = y_under_[y_under_==under[0]]
        else:
            dict_under = dict(zip(under, [n for i in range(len(under))]))
            under_sam = RandomUnderSampler(sampling_strategy=dict_under)
            is_under = np.in1d(y, under)
            X_under, y_under = under_sam.fit_resample(X[is_under], y[is_under])

        X_combined_sampling = np.concatenate((X_over, X_under))
        y_combined_sampling = np.concatenate((y_over, y_under))
        return X_combined_sampling, y_combined_sampling


# Selects run based on combination
def select_run(results, missing):
    """
    Helper function selecting run based on missing classes
    :param results: dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param missing: list of missing classes
    :return: subset of predictions for the chosen combination
    """
    n_missing = len(missing)
    for i in range(n_missing):
        results = results[results[("Missing " + str(i + 1))].isin(missing)]
    return results


# Creates result file
def get_all(results, to_csv, PATH_out):
    """
    Calculates all perfomance scores for each masked combination
    :param results: dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param to_csv: whether to write csv
    :param PATH_out: path to write to
    :return:
    """
    n_missing = results.shape[1] - 4
    unique_classes = set(results["y_true"])
    combs = list(combinations(unique_classes, n_missing))
    cols = ["Missing", "alpha", "H", "Acc_known", "Acc_unknown", "H_semi", "Acc_known_semi", "Acc_unknown_semi"]
    missing = []
    alpha = []
    h_list = []
    known_list = []
    unknown_list = []
    h_list_semi = []
    known_list_semi = []
    unknown_list_semi = []
    for i in combs:
        selected = select_run(results, list(i))
        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"), selected["y_pred"].to_numpy(dtype="int32"), list(i))
        missing.append(str(i))
        alpha.append(selected["alpha"].iloc[0])
        h_list.append(h)
        known_list.append(acc_known)
        unknown_list.append(acc_unknown)

        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"),
                                            selected["y_pred_semi"].to_numpy(dtype="int32"), list(i))
        h_list_semi.append(h)
        known_list_semi.append(acc_known)
        unknown_list_semi.append(acc_unknown)

    scores = pd.DataFrame(zip(missing, alpha, h_list, known_list, unknown_list,
                              h_list_semi, known_list_semi, unknown_list_semi), columns=cols)
    if to_csv:
        scores.to_csv(PATH_out)
    return scores


def plot_latent(z_source, source_labels, z_target, hy_test, pred, remove_col, filename):
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, figsize=(3, 12), sharex=True, sharey=True)

    #print(z_source.shape) #11*3300
    latent_space = TSNE(n_components=2) 
    latent_df = latent_space.fit_transform(np.concatenate([np.array(z_source.T), np.array(z_target.T)],axis=0))
    #print(latent_df.shape) # 4323 *2
    z_source = latent_df[:len(z_source.T),:].T
    z_target = latent_df[len(z_source.T):,:].T

    df=pd.DataFrame()
    df['axis1'] = z_source[0,:]#latent_df[:,0]
    df['axis2'] = z_source[1,:]#latent_df[:,1]
    df['label'] = [str(x) for x in source_labels]
    df['batch'] = None
    #latent_df2 = latent_space.fit_transform(z_target.T)
    df2=pd.DataFrame()
    df2['axis1'] = z_target[0,:]#latent_df2[:,0]
    df2['axis2'] = z_target[1,:]#latent_df2[:,1]
    df2['label'] = [str(x) for x in hy_test]
    df2['pred'] = [str(x) for x in pred]
    df2['batch'] = [0 if x not in remove_col else 1 for x in hy_test]

    sns.scatterplot(
        x="axis1", y="axis2",
        size=3,
        sizes=(3,10),
        data=df,
        legend="full",
        hue='label',
        hue_order=[str(x) for x in sorted(set(source_labels))],
        alpha=0.6,
        linewidth=(0),
        edgecolors='none',
        ax=ax0,
        marker='o',
        )
    sns.scatterplot(
        x="axis1", y="axis2",
        size=3,
        sizes=(3,10),
        data=df2,
        legend="full",
        hue='batch',
        hue_order=[0,1],
        alpha=0.6,
        linewidth=(0),
        edgecolors='none',
        ax=ax1,
        marker='o',
        )
    sns.scatterplot(
        x="axis1", y="axis2",
        size=3,
        sizes=(3,10),
        data=df2,
        legend="full",
        hue='pred',
        hue_order=[str(x) for x in sorted(set(hy_test))],
        alpha=0.6,
        linewidth=(0),
        edgecolors='none',
        ax=ax2,
        marker='o',
        )
    sns.scatterplot(
        x="axis1", y="axis2",
        size=3,
        sizes=(3,10),
        data=df2,
        legend="full",
        hue='label',
        hue_order=[str(x) for x in sorted(set(hy_test))],
        alpha=0.6,
        linewidth=(0),
        edgecolors='none',
        ax=ax3,
        marker='o',
        )


    ax0.legend(bbox_to_anchor=(1.04,0.00), loc='lower left')
    ax1.legend(bbox_to_anchor=(1.04,0.50), loc='lower left')
    ax2.legend(bbox_to_anchor=(1.04,0.00), loc='lower left')
    ax3.legend(bbox_to_anchor=(1.04,0.00), loc='lower left')
    plt.savefig(filename, bbox_inches='tight')

    return
