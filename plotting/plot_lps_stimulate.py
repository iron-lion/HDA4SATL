import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score



def plot_(args):
    file_list = []
    searchstr = os.path.join(
                f'./results/mnn_lps_pca_50/',
                f'*.json')

    files = glob.glob(searchstr)
    file_list.extend(files)

    content2 = pd.DataFrame(columns=[])

    for filename in file_list:
        print(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            content2 = content2.append(
                pd.DataFrame({
                    'filename': [filename],
                    'train_site': [filename.split('_')[-7].split('-')[0]],
                    'test_site': [filename.split('_')[-7].split('-')[1]],
                    'seed': [filename.split('_')[-1].rstrip('.json')],
                    }),
                ignore_index=True,
                )
    pd.options.display.max_rows = 999
    print(content2)
    content_ = content2.copy()

    values = [pd.DataFrame(columns=[])] * 1

    for tr_site in ['mouse1','mouse2','mouse3','rat1','rat2','rat3','pig1','pig2','pig3','rabbit1','rabbit2','rabbit3']:
        for te_site in ['mouse1','mouse2','mouse3','rat1','rat2','rat3','pig1','pig2','pig3','rabbit1','rabbit2','rabbit3']:
            content_scenario = content_.query("train_site==@tr_site")
            content_scenario = content_scenario.query("test_site==@te_site")

            if content_scenario.shape[0]>0:

                results = []
                for filename in content_scenario['filename'].values:
                    with open(filename) as f:
                        data = json.load(f)
                        results.append(data['h'])
                result_mean_all = round(np.mean(results), 3)
                result_std_all = round(np.std(results), 3)

                values[0] = values[0].append(
                    pd.DataFrame({
                        'train_site': [tr_site],
                        'test_site': [te_site],
                        'result': [result_mean_all],
                        'result_std_all': [result_std_all],
                        }),
                    ignore_index=True,
                    sort=False,
                    )
            else:
                values[0] = values[0].append(
                    pd.DataFrame({
                        'train_site': [tr_site],
                        'test_site': [te_site],
                        'result': [np.nan],
                        'result_std_all': [np.nan],
                        }),
                    ignore_index=True,
                    sort=False,
                    )

    # -------------
    print(f'plotting.. {args.outfile}')
    rc = {
            'legend.fontsize': 10,
            'axes.labelsize': 12,
            'ytick.labelsize': 13,
            'xtick.labelsize': 13,
          }

    plt.close('all')
    sns.set(style="whitegrid",
            rc=rc,
            font_scale=1.1
    )
   
    fig, ax = plt.subplots(1, 1, figsize=(9,8),dpi=300)
    for i in range(1):
        # heatmap 
        matrix = values[i].pivot("train_site", "test_site", "result")
        matrix = matrix[['mouse1','mouse2','mouse3','rat1','rat2','rat3','pig1','pig2','pig3','rabbit1','rabbit2','rabbit3']]
        matrix = matrix.reindex(['mouse1','mouse2','mouse3','rat1','rat2','rat3','pig1','pig2','pig3','rabbit1','rabbit2','rabbit3'])
        print(matrix)
        sns.heatmap(matrix.T, 
                    annot=True, 
                    fmt=".2f",
                    linewidths=.90,
                    vmin=0.5, 
                    vmax=1,
                    ax=ax,
                    cmap=sns.cubehelix_palette(8, start=.5, rot=-.75),
                    cbar=False,
                    cbar_kws={'label': 'AUROC'}
        )

        # adjust y axis label position
        yticks = ax.get_yticks()
        ax.set_yticks([i for i in yticks])
        ylabs = ax.get_xticklabels()
        ax.set_yticklabels(ylabs)

        xticks = ax.get_xticks()
        ax.set_xticks([i+0.15 for i in xticks])
        ax.set_xticklabels(ylabs)

        ax.set_ylabel('Source')    
        ax.set_xlabel('Target')

    # adjust one yticks explicitly, as they are not covered
    # by the command above
    plt.subplots_adjust(wspace=0.01)
    fig.autofmt_xdate(rotation=45)
    fig.colorbar(ax.get_children()[0], ax=ax)
    plt.margins(0.01)
    plt.title('PCA50 feature - MNN', fontsize = 24)
    plt.savefig(f'./plots/{args.outfile}_lps_norm_mnn_pca.png')
    plt.savefig(f'./plots/{args.outfile}_lps_norm_mnn_pca.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric',
                        type=str,
                        default='auroc')
    parser.add_argument('--outfile',
                        type=str,
                        default='cell_stimulation_study')
    args = parser.parse_args()

    plot_(args)
