import os
import sys
from datetime import datetime
import random
import pandas as pd
import numpy as np
import logging
import json
import argparse
import h5py
from src.satl import *
from src.data_loader import *
from src.utils import *

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0,"./adapt")
from adapt.feature_based import TCA
from adapt.feature_based import fMMD
from adapt.instance_based import IWC
from adapt.instance_based import LDM
from adapt.instance_based import KMM
from adapt.instance_based import TrAdaBoost
from adapt.instance_based import BalancedWeighting as BW
from adapt.instance_based import NearestNeighborsWeighting as NNW
from adapt.feature_based import FA
from adapt.feature_based import SA
from adapt.feature_based import CORAL
from adapt.feature_based import PRED
from adapt.parameter_based import TransferTreeClassifier
from adapt.parameter_based import TransferForestClassifier


def main(seed, args):
    seed = scenario['seed']
    thisK = scenario['thisK']
    oversample = scenario['oversample']
    ndim = scenario['ndim']
    num_to_del = scenario['num_to_del']
    pca_dim = scenario['pca_dim']
    model = scenario['model']
    name_model = scenario['name_model']

    random.seed(seed)
        
    path = f'./results/adapt_baron_norm/{name_model}_mouse-human_pca_{pca_dim}_numdel_{num_to_del}_seed_{seed}.json'
    if os.path.exists(path):
        logging.info(f'{path}')
        return

    root_dir = './dataset/baron_norm/'
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

        if name_model == 'TransferTree':
            s_model = DecisionTreeClassifier().fit(dataset['X_source'], dataset['y_source'])
            model = TransferTreeClassifier(s_model)

            model.fit(X_seen, y_seen)

        elif name_model == 'TransferForest':
            s_model = RandomForestClassifier().fit(dataset['X_source'], dataset['y_source'])
            model = TransferForestClassifier(s_model)

            model.fit(X_seen, y_seen)

        else:
            model.fit(dataset['X_source'], dataset['y_source'], X_seen, y_seen)
            
        pred = model.predict(dataset['X_test'])
        h, acc_known, acc_unknown = h_score(dataset['y_test'], pred, remove_col)
        print(name_model, h, acc_known, acc_unknown)

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
    scenarios.append({  'name_model': 'TCA', 'model': TCA(RidgeClassifier(), n_components = 10, mu=1, kernel="rbf", gamma=0.1, verbose=0, random_state=0),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'TransferTree', 'model': None,
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'TransferForest', 'model': None,
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'LDM', 'model': LDM(RidgeClassifier()),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'PRED', 'model': PRED(RidgeClassifier(0.), pretrain=True, random_state=0),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'CORAL', 'model': CORAL(RidgeClassifier(), random_state=0),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'SA', 'model': SA(RidgeClassifier()),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'FA', 'model': FA(RidgeClassifier()),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'TrAdaBoost', 'model': TrAdaBoost(RidgeClassifier(), n_estimators=10),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'BW', 'model': BW(RidgeClassifier(), gamma=0.5),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'NNW', 'model': NNW(RidgeClassifier(), n_neighbors=5),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'KMM', 'model': KMM(RidgeClassifier(), kernel="rbf", gamma=1.),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'fMMD', 'model': fMMD(RidgeClassifier(), kernel="linear"),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    scenarios.append({  'name_model': 'IWC', 'model': IWC(RidgeClassifier(0.), classifier=RidgeClassifier(0.), random_state=0),
                        'seed': 7, 'thisK': 10, 'oversample': 300, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})

    for scenario in scenarios:
        main(num_cores, scenario)
