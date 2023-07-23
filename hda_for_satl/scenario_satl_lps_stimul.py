import os, random
import logging
import json
import gc
from datetime import datetime
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
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

    root_dir = './dataset/norm_pca50/norm_pca50_'

    out_dir = f'./results/satl_lps_pca_50/lps_stim_{source_species}-{target_species}_pca_{pca_dim}_numdel_{num_to_del}_seed_{seed}'
    if os.path.exists(out_dir + '_pred.csv'):
        logging.info(f'Skip: {out_dir}_pred.csv')
        return
    else:
        logging.info(f'Run:  {out_dir}_pred.csv')

    latent_space = None# PCA(n_components=pca_dim)
    dataset = lps_stimulate_load(str(root_dir), str(source_species), str(target_species), latent_space)
    
    run_all = SATL(dataset,
            out_dir,
            num_to_del, dim=ndim, alpha=[0.01, 0.1, 1, 10, 100],
            n_resample_source=oversample, n_resample_target=oversample, n_jobs = n_jobs, K=thisK)
    run_all.run_mode(False)

    del(run_all)
    del(dataset)
    gc.collect()


if __name__ == '__main__':
    jobid = os.getenv('SLURM_JOB_ID')

    timestamp = datetime.now().timestamp()
    date_time = datetime.fromtimestamp(timestamp)
    current_time = date_time.strftime("%d%m%y_%H%M%S")

    logger = create_logger(jobid, current_time)
    
    try:
        num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except:
        num_cores = 1
    logger.info(f'#{num_cores} cores available')


    scenarios = []
    for pair in list(itertools.permutations(['mouse','pig','rat','rabbit'], 2)):
        for idx in list(itertools.product([1,2,3], repeat=2)):
            source_species = pair[0]+str(idx[0])
            target_species = pair[1]+str(idx[1])
            scenarios.append({'source_species': str(source_species),
                                'target_species': str(target_species),
                                'seed': 37,
                                'thisK': 10,
                                'oversample': 300,
                                'ndim': 4,
                                'num_to_del': 1,
                                'pca_dim': 50})

    #from multiprocessing import set_start_method
    #set_start_method("spawn")

    for scenario in scenarios:
        main(num_cores, scenario)
