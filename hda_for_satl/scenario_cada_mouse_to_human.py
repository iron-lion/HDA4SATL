import os
import glob
import json
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from src.model_cada import CAE
from src.data_loader import *
from src.satl import *


def main(seed, args):
    root_dir = "./dataset/datasets/"
    organ = 'bm'
    prep = 'scanpy'
    dataset = preset_mousehuman_load(root_dir,  organ, prep)

    results = {
        'seed' : seed,
        'organ' : organ,
        'feature' : prep,
        'h':[],
        'acc_known':[],
        'acc_unknown':[],
    }
    
    combs = list(combinations(list(set(dataset['y_source'])), 2))
    for remove_col in combs:
        data_loader = GZSL_data_loader(dataset, remove_col)

        model = CAE(args, dataset['X_train'].shape[1], dataset['X_source'].shape[1], [50, 50], [50,50], 50, bn=False, relu=0.0)

        model.init_model()
        model.train_vae(data_loader)
 
        h, acc_known, acc_unknown = model.train_classifier()       
        
        results['h'].append(h)
        results['acc_known'].append(acc_known)
        results['acc_unknown'].append(acc_unknown)
        print(f'best_ {remove_col} target: {h}\t{acc_known}\t{acc_unknown}')


    results['h'] = str(np.mean(results['h']))
    results['acc_known'] = str(np.mean(results['acc_known']))
    results['acc_unknown'] = str(np.mean(results['acc_unknown']))
    print(results)

    # Serializing json
    json_object = json.dumps(results, default=lambda o: o.__dict__, sort_keys=True, indent = 4)
    with open(f'./results/cada_clf/CADA_organ_{organ}_feature_{prep}_seed_{seed}.json', "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-nbatch','--batch_size',
                        type = int,
                        default = 10,
                        help='training batch_size')
       
    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=200000)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20) 

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
     
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')
    
    parser.add_argument('--save',
                        action='store_true',
                        help='training network saving')
   
    args = parser.parse_args()
    args.device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    print(args)

    for seed in [12,436,7457,113,346,2,67,8,70,16]:
        main(seed, args)
