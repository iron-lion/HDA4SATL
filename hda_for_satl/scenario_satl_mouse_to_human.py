import os, random
import glob
import logging
import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import src.utils as utils
from src.data_loader import *
from src.satl import *


def main(n_jobs:int, scenario:dict):
    organ = scenario['organ']
    prep = scenario['prep']
    seed = scenario['seed']
    thisK = scenario['thisK']
    oversample = scenario['oversample']
    ndim = scenario['ndim']

    random.seed(seed)
    root_dir = "./dataset/"
    dataset = preset_mousehuman_load(root_dir, organ, prep)

    out_dir = f'satl_mouse_to_human/MouseToHuman_{organ}_{prep}_seed_{seed}_K_{thisK}_over_{oversample}_ndim_{ndim}'

    if os.path.exists('./results/' + out_dir + '_pred.csv'):
        print('skip')
        return


    run_all = SATL(dataset,
                    out_dir,
                    2, 
                    dim=ndim, 
                    alpha=[0.01, 0.1, 1, 10, 100],
                    n_resample_source=oversample, n_resample_target=oversample, n_jobs = n_jobs, K=thisK, feature_analysis=False)
    run_all.run_mode()


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

    #from multiprocessing import set_start_method
    #set_start_method("spawn")
    #logger.info('Starting pooling')

    scenarios = [
        {'organ': 'brain', 'prep': 'scetm',    'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'brain', 'prep': 'scanpy',   'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'brain', 'prep': 'dca',      'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'bm', 'prep': 'scanpy',   'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'bm', 'prep': 'scetm',    'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'bm', 'prep': 'dca',      'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'pancreas', 'prep': 'scanpy',   'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'pancreas', 'prep': 'scetm',    'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
        {'organ': 'pancreas', 'prep': 'dca',      'seed': 37, 'thisK': 10, 'oversample': -1, 'ndim': 11},
    ]

    for scenario in scenarios:
        main(num_cores, scenario)
