import os, random
import glob
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import mnnpy
from src.satl import *
from src.data_loader import *
from src.utils import *


def main(n_jobs:int, scenario:dict):
    seed = scenario['seed']
    thisK = scenario['thisK']
    oversample = scenario['oversample']
    ndim = scenario['ndim']
    num_to_del = scenario['num_to_del']
    pca_dim = scenario['pca_dim']

    random.seed(seed)
    
    path = f'./results/mnn_baron_norm/baron_mnn_mouse-human_pca_{pca_dim}_numdel_{num_to_del}_seed_{seed}.json'
    if os.path.exists(path):
        logging.info(f'{path}')
        return

    root_dir = './dataset/'
    latent_space = PCA(n_components=pca_dim)
    dataset, label_set = baron_load(root_dir, latent_space)
    
    results = {
        'masked cell number' : num_to_del,
        'seed' : seed,
        'h' : [],
        'acc_known' : [],
        'acc_unknown' : [],
    }
    
    combs = list(combinations(list(label_set), num_to_del))
    for remove_col in combs:
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(dataset['X_train'], dataset['y_train'], masked_cells=remove_col)

        corrected = mnnpy.mnn_correct(dataset['X_source'], X_seen, dataset['X_test'], var_index = range(0,50), batch_categories = ["mouse", "human_seen", "human_test"])
        adata = corrected[0]

        mnn_mouse_data = adata[:len(dataset['X_source']),:]
        mnn_X_seen = adata[len(dataset['X_source']): len(dataset['X_source'])+len(X_seen),:]
        mnn_hX_test = adata[len(dataset['X_source']) + len(X_seen):,:]
       
        model = RandomForestClassifier().fit(mnn_mouse_data, dataset['y_source'])
        pred = model.predict(mnn_hX_test)
        h, acc_known, acc_unknown = h_score(dataset['y_test'], pred, remove_col)
        print(h, acc_known, acc_unknown)

        results['h'].append(h)
        results['acc_known'].append(acc_known)
        results['acc_unknown'].append(acc_unknown)
           
    results['h_std'] = np.std(results['h'])
    results['h'] = np.mean(results['h'])
    results['acc_known_std'] = np.std(results['acc_known'])
    results['acc_known'] = np.mean(results['acc_known'])
    results['acc_unknown_std'] = np.std(results['acc_unknown'])
    results['acc_unknown'] = np.mean(results['acc_unknown'])
    
    # Serializing json
    json_object = json.dumps(results, default=lambda o: o.__dict__, sort_keys=True, indent = 4)
    with open(path, "w") as outfile:
        outfile.write(json_object)
    

if __name__ == '__main__':
    jobid = os.getenv('SLURM_JOB_ID')

    timestamp = datetime.now().timestamp()
    date_time = datetime.fromtimestamp(timestamp)
    current_time = date_time.strftime("%d%m%y_%H%M%S")

    # log configuration 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    #logging.FileHandler(f'./job_{jobid}_time_{current_time}.log')

    # XXX: Not used here
    try:
        num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except:
        num_cores = 1
    logger.info(f'#{num_cores} cores available')


    scenarios = []
    scenarios.append({'seed': 37,
                        'thisK': 10,
                        'oversample': 300,
                        'ndim': 11,
                        'num_to_del': 2,
                        'pca_dim': 50})

    for scenario in scenarios:
        main(num_cores, scenario)
