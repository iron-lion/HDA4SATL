import os, random
import glob
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    
    out_dir = f'./satl_baron_norm/baron_satl_mouse-human_pca_{pca_dim}_numdel_{num_to_del}_seed_{seed}.json'
    if os.path.exists(out_dir):
        logging.info(f'{out_dir}')
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
 

    run_all = SATL(dataset,
                    out_dir,
                    2, 
                    dim=ndim, 
                    alpha=[0.01, 0.1, 1, 10, 100],
                    n_resample_source=oversample, n_resample_target=oversample, n_jobs = n_jobs, K=thisK, feature_analysis=True)
    run_all.run_mode()

   

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
    logging.FileHandler(f'./satl_baron_job_{jobid}_time_{current_time}.log')

    # XXX: Not used here
    try:
        num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except:
        num_cores = 1
    logger.info(f'#{num_cores} cores available')


    scenarios = []
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 2})
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 11})
    scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 50})
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 100})
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 200})
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 400})
    #scenarios.append({'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11, 'num_to_del': 2, 'pca_dim': 800})


    for scenario in scenarios:
        main(num_cores, scenario)
