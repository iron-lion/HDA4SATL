import os, random
import logging
import glob
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import seaborn as sns
import anndata as ad
import scanpy as sc
sc.settings.verbosity = 3

def _get_dataframe(flist):
    tb = pd.DataFrame([])
    for f in flist:
        v = pd.read_csv(f, sep=' ', header=0, index_col=0)
        tb = pd.concat([tb, v], axis=1)
    tb = tb.transpose()
    return tb


def _get_data(mo):
    dir_path = './dataset/bm-6754/'
    dataset_unst_files = glob.glob(dir_path + mo + '*unst*.txt')
    dataset_lps2_files = glob.glob(dir_path + mo + '*lps2*.txt')
    dataset_lps4_files = glob.glob(dir_path + mo + '*lps4*.txt')
    dataset_lps6_files = glob.glob(dir_path + mo + '*lps6*.txt')
    
    s_unst = _get_dataframe(dataset_unst_files)
    s_lps2 = _get_dataframe(dataset_lps2_files)
    s_lps4 = _get_dataframe(dataset_lps4_files)
    s_lps6 = _get_dataframe(dataset_lps6_files)

    dataset = pd.concat([s_unst, s_lps2, s_lps4, s_lps6], axis=0)
    labels = [0] * len(s_unst) + [1] * len(s_lps2) + [2] * len(s_lps4) + [3] * len(s_lps6)
    del(s_unst, s_lps2, s_lps4, s_lps6)
    
    adata = ad.AnnData(dataset, dtype=np.int32)
    adata.obs["label"] = labels
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    adata = adata[adata.obs.n_genes < 4000, :]

    
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, n_comps=50)
    mtx = adata.obsm['X_pca']
    print()
    
    adata_subset = adata[adata.obs['label']==0]
    print('sample0', len(adata_subset.obs_names))
    print('gene', len(adata_subset.var_names))
    final_data = pd.DataFrame(adata_subset.obsm['X_pca'])
    final_data = final_data.T
    final_data.to_csv(f'norm_pca50_{mo}_unst.csv')

    adata_subset = adata[adata.obs['label']==1]
    print('sample1', len(adata_subset.obs_names))
    print('gene', len(adata_subset.var_names))
    final_data = pd.DataFrame(adata_subset.obsm['X_pca'])
    final_data = final_data.T
    final_data.to_csv(f'norm_pca50_{mo}_lps2.csv')
    
    adata_subset = adata[adata.obs['label']==2]
    print('sample2', len(adata_subset.obs_names))
    print('gene', len(adata_subset.var_names))
    final_data = pd.DataFrame(adata_subset.obsm['X_pca'])
    final_data = final_data.T
    final_data.to_csv(f'norm_pca50_{mo}_lps4.csv')

    adata_subset = adata[adata.obs['label']==3]
    print('sample3', len(adata_subset.obs_names))
    print('gene', len(adata_subset.var_names))
    final_data = pd.DataFrame(adata_subset.obsm['X_pca'])
    final_data = final_data.T
    final_data.to_csv(f'norm_pca50_{mo}_lps6.csv')
    return adata


_get_data('rabbit1')
_get_data('rabbit2')
_get_data('rabbit3')
_get_data('pig1')
_get_data('pig2')
_get_data('pig3')
_get_data('rat1')
_get_data('rat2')
_get_data('rat3')
_get_data('mouse1')
_get_data('mouse2')
_get_data('mouse3')

