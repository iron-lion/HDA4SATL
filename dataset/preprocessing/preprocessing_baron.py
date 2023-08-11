import os, random
import glob
import json
import sys
import pandas as pd
import numpy as np
import utils as utils
from sklearn import preprocessing
from utils import h5_data_loader

#import scanpy.api as sc
import anndata as ad
import scanpy as sc

def main():
    label_filter = None
    human_data, human_labels, _, _ = h5_data_loader(["./dataset/baron/baron_sc.h5"], label_filter)
    mouse_data, mouse_labels, _, _ = h5_data_loader(["./dataset/baron/baron_mouse_sc.h5"], label_filter)
    print('rawpca')
    print('m', mouse_data.shape)
    print('h', human_data.shape)

    common_set = set(human_labels) & set(mouse_labels)
    all_set = set(human_labels) | set(mouse_labels)
    out_set = all_set - common_set
    le = preprocessing.LabelEncoder()
    le.fit(list(common_set))
    
    keep = np.in1d(human_labels, list(common_set), invert=False)
    human_data = human_data.loc[keep,:]
    human_labels = human_labels[keep]
    del(keep)
    keep = np.in1d(mouse_labels, list(common_set), invert=False)
    mouse_data = mouse_data.loc[keep,:]
    mouse_labels = mouse_labels[keep]
    del(keep)

    adata = ad.AnnData(human_data, dtype=np.int32)
    adata.obs["label"] = human_labels
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    adata = adata[adata.obs.n_genes < 4000, :]

    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    print('gene', len(adata.obs_names))
    print('gene', len(adata.var_names))
    final_data = pd.DataFrame(adata.X, columns=adata.var_names)
    final_label = pd.DataFrame(adata.obs["label"])
    final_data = final_data.T
    final_data.to_csv(f'./dataset/baron_norm/norm_human.csv')
    final_label.to_csv(f'./dataset/baron_norm/norm_human_label.csv')
    print(final_data)

    adata = ad.AnnData(mouse_data, dtype=np.int32)
    adata.obs["label"] = mouse_labels
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=5)
    adata = adata[adata.obs.n_genes < 4000, :]

    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    print('gene', len(adata.obs_names))
    print('gene', len(adata.var_names))
    final_data = pd.DataFrame(adata.X, columns=adata.var_names)
    final_label = pd.DataFrame(adata.obs["label"])
    final_data = final_data.T
    final_data.to_csv(f'./dataset/baron_norm/norm_mouse.csv')
    final_label.to_csv(f'./dataset/baron_norm/norm_mouse_label.csv')
    print(final_data)
    exit()



if __name__ == '__main__':
    main()
