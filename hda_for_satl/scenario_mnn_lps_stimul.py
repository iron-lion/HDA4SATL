import os, random
import logging
import glob
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from lpproj import LocalityPreservingProjection  as LPP
import mnnpy
import itertools
from src.satl import *
from src.data_loader import *
from src.utils import *



def main(n_jobs:int, scenario:dict):
    source_species = scenario['source_species']
    target_species = scenario['target_species']
    seed = scenario['seed']
    thisK = scenario['thisK']
    oversample = scenario['oversample']
    ndim = scenario['ndim']
    num_to_del = scenario['num_to_del']
    pca_dim = scenario['pca_dim']

    random.seed(seed)

    root_dir = './dataset/norm_'
    #latent_space = LPP(n_components=pca_dim)
    latent_space = PCA(n_components=pca_dim)
    dataset = lps_stimulate_load(str(root_dir), str(source_species), str(target_species), latent_space)

    path = f'./results/mnn_lps_pca_50/mnn_lps_norm_{source_species}-{target_species}_pca_{pca_dim}_numdel_{num_to_del}_seed_{seed}.json'
    if os.path.exists(path):
        loggine.info(f'... result file exists ... {path}')
        return
    logging.info(f'{path}')

    results = {
        'masked cell number' : num_to_del,
        'seed' : seed,
        'h' : [],
        'acc_known' : [],
        'acc_unknown' : [],
    }
    
    combs = list(combinations([0,1,2,3], num_to_del))
    for remove_col in combs:
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(dataset['X_train'], dataset['y_train'], masked_cells=remove_col)

        corrected = mnnpy.mnn_correct(dataset['X_source'], X_seen, dataset['X_test'], var_index = range(0,pca_dim), batch_categories = ["source", "target_seen", "target_test"], k=50)
        adata = corrected[0]

        mnn_source = adata[:len(dataset['X_source']),:]
        mnn_X_seen = adata[len(dataset['X_source']): len(dataset['X_source'])+len(X_seen),:]
        mnn_hX_test = adata[len(dataset['X_source']) + len(X_seen):,:]
       
        model = RandomForestClassifier().fit(mnn_source, dataset['y_source'])
        pred = model.predict(mnn_hX_test)
        h, acc_known, acc_unknown = h_score(dataset['y_test'], pred, remove_col)

        conf_matrix = confusion_matrix(dataset['y_test'], pred)
        logging.info(f'{remove_col} {h} {acc_known} {acc_unknown} {conf_matrix}')
        results['h'].append(h)
        results['acc_known'].append(acc_known)
        results['acc_unknown'].append(acc_unknown)

    results['h_std'] = np.std(results['h'])
    results['h_ori'] = results['h']
    results['h'] = np.mean(results['h'])
    results['acc_known_std'] = np.std(results['acc_known'])
    results['acc_known_ori'] = results['acc_known']
    results['acc_known'] = np.mean(results['acc_known'])
    results['acc_unknown_std'] = np.std(results['acc_unknown'])
    results['acc_unknown_ori'] = results['acc_unknown']
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
    for idx in list(itertools.product([1,2,3], repeat=2)):
        for pair in list(itertools.permutations(['mouse','pig','rat','rabbit'], 2)):
            source_species = pair[0]+str(idx[0])
            target_species = pair[1]+str(idx[1])
            scenarios.append({'source_species': str(source_species),
                                'target_species': str(target_species),
                                'seed': 37,
                                'thisK': 10,
                                'oversample': 0,
                                'ndim': 4,
                                'num_to_del': 1,
                                'pca_dim': 50})

    for scenario in scenarios:
        main(num_cores, scenario)
