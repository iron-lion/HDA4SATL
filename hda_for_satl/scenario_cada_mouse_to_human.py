import os
import glob
import json
import sys
import copy
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from src.model_cada import CAE
from src.data_loader import *
from src.utils import *
from src.satl import *


def main(args, prefixname):
    root_dir = "./dataset/"
    organ = args['organ']
    prep =  args['prep']
    dataset = preset_mousehuman_load(root_dir,  organ, prep)

    results = {
        'seed' : args["seed"],
        'organ' : organ,
        'feature' : prep,
        'h':[],
        'acc_known':[],
        'acc_unknown':[],
    }
    
    combs = list(combinations(list(set(dataset['y_source'])), 2))
    for remove_col in combs:
        data_loader = GZSL_data_loader(dataset, remove_col)

        model = CAE(args, dataset['X_train'].shape[1], dataset['X_source'].shape[1], [50], [50], 11, bn=False, relu=0.2)

        model.init_model()
        model.train_vae(data_loader)
 
        h, acc_known, acc_unknown = model.train_classifier()       
        model.plot_loss(f'{prefixname}_{organ}_{prep}_{remove_col[0]}_{remove_col[1]}_{h:.3f}_{acc_known:.3f}_{acc_unknown:.3f}.png')

        results['h'].append(h)
        results['acc_known'].append(acc_known)
        results['acc_unknown'].append(acc_unknown)
        logger.info(f'best_ {remove_col} target: {h}\t{acc_known}\t{acc_unknown}')


    results['h'] = str(np.mean(results['h']))
    results['acc_known'] = str(np.mean(results['acc_known']))
    results['acc_unknown'] = str(np.mean(results['acc_unknown']))
    logger.info(results)

    # Serializing json
    json_object = json.dumps(results, default=lambda o: o.__dict__, sort_keys=True, indent = 4)
    with open(f'./{prefixname}_{organ}_{prep}_seed_{args["seed"]}.json', "w") as outfile:
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

    scenarios = []
    __m = 1
    __v = 2
    scenarios.append({'organ':'pancreas', 'prep':'scanpy',
                        'seed': 7,
                        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                        'batch_size': 32,
                        'epochs': 100*__v,
                        'learning_rate': 0.002,
                        'lr_scheduler_step': 80,
                        'lr_scheduler_gamma': 0.7,
                        'beta': {'weight':0.00005, 'start': 0, 'end': 97*__m}, # 0.5
                        'cross_recon': {'weight': 4.37, 'start': 21*__m, 'end': 97*__m}, # 2.37
                        'dist': {'weight': 8.13, 'start': 6*__m, 'end': 70*__m}, # 8.13
                        })
    
    
    for i in range(1,1):
        new_scenario = copy.deepcopy(scenarios[0])
        new_scenario['learning_rate'] = scenarios[i-1]['learning_rate']*2
        scenarios.append(new_scenario)
    

    for sc in scenarios:
        main(sc, f"T2_400_d_{sc['dist']['weight']}_cr_{sc['cross_recon']['weight']}_b_{sc['beta']['weight']}_lr_{sc['learning_rate']}")
