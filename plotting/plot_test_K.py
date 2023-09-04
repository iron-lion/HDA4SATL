import os
import json
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from warnings import simplefilter

def _plot(args):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    PATH_fig4 = os.path.join('./results/satl_testK/')

    # create dataframe giving an overview of all files in path
    file_list = []
    for (_, _, filenames) in os.walk(PATH_fig4):
        [file_list.append(f) for f in filenames if ('_K_' in f) and ('_h' in f)]
        break

    content = pd.DataFrame(columns=['dim',
                                    'organ',
                                    'prep',
                                    'h',
                                    'h_std',
                                    ])

    for filename in file_list:
        with open(PATH_fig4 + filename) as f:
            data = pd.read_csv(f, sep=',')

            dim = int(filename.split('_')[4])
            organ = str(filename.split('_')[0])
            prep = str(filename.split('_')[9])

            content = content.append(
                pd.DataFrame({
                    'dim': [dim],
                    'organ' : [organ],
                    'prep' : [prep],
                    'h': np.mean(data['H_semi']),
                    'h_std': np.std(data['H_semi']),
                    }),
                ignore_index=True,
                )
    #print(content)
    
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    c = content.groupby(['organ', 'prep'])
    for k, curve in c:
        print(k)
        nc = curve.groupby('dim').agg({'h': [np.mean, np.std]})
        #curve = pd.DataFrame(curve)
        #curve['dim'].astype('int')
        #curve['h'].astype('float')
        #curve['h_std'].astype('float')
        #curve = curve.sort_values(by=['dim'])
        #print(curve['h'].tolist())
        #curve.fillna()
        # ------------
        # plot
        # ------------
        fm = k[1] if k[1] != 'scanpy' else 'seurat'
        ax.plot(nc.index, nc['h']['mean'], lw=2, label=k[0]+'-'+fm)
        ax.fill_between(nc.index, nc['h']['mean']+nc['h']['std'], nc['h']['mean']-nc['h']['std'], alpha=0.4)

    """
    # ------------
    # axes limits and labels
    # ------------
    xy_label_fs = 20
    #ax.set_xlabel('Number of unknown classes', fontsize=xy_label_fs)
    ax.set_xlim([-0.01, 11.01])
    ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels(['1/10','2/9','3/8','4/7','5/6','6/5','7/4','8/3','9/2','10/1'], size=20)
    ax.set_ylim([0.0, 1.01])
    """
    ax.legend(bbox_to_anchor=(1.01, 0.01), loc='lower left',
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})
    ax.set_xlabel('K', fontsize=20)
    ax.set_ylabel('h-score', fontsize=20)


    plt.tight_layout()
    plt.savefig(f'./plots/satl_K_test.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile',
                        type=str,
                        default='ablation')
    args = parser.parse_args()

    _plot(args)
