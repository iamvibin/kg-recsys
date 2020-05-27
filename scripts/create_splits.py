import os
import random
import pandas as pd
import argparse
from data_reformat import reformat
import json
from preprocess import preprocess
from preprocess import generate_similar_pair
import numpy as np
from scipy.stats import wishart
from BPMF.recommend.bpmf import BPMF
from BPMF.recommend.utils.evaluation import RMSE
from run_bpmf import run_baseline
from run_bpmf import get_baseline_output
from sim_interactions_from_kg import generate_user_item_pair
from generate_blocking_target import generate_blocking_target
from src.main import startBaseline

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='book', help='which dataset to use')
parser.add_argument('--n', type=int, default=5, help='the number of neighbors')
parser.add_argument('--t', type=float, default=0.0, help='threshold')
parser.add_argument('--s', type=int, default=555, help='seed value')
parser.add_argument('--i', type=int, default=0, help='running on split')
parser.add_argument('--out', type=str, default='000', help='default directory')
parser.add_argument('--list', type=list, default=[], help='list of neighbours')
parser.add_argument('--type', type=str, default='eval', help='list of neighbours')
parser.add_argument('--n_items', type=int, default=0, help='number of entities')
parser.add_argument('--n_users', type=int, default=0, help='number of users')
args = parser.parse_args()
EVAL = 'eval'
TRAIN = 'train'
eval_types = [EVAL, TRAIN]
splits = 1
seed_list = random.sample(range(0, 2**32-1), splits)
#seed_list = [1238879808, 2790790704, 1154929626, 867958829, 2920458158, 865877606, 1056154238, 2144527568, 1464353066, 3087242964]
path = os.getcwd()
data_path = os.path.join(path, '..', 'data', args.d)

neighbours = [2]
print('Creating splits for the baseline....')

entity_id2index ,relation_id2index, item_index_old2new, user_cnt = reformat(args)
args.n_items = len(entity_id2index)
args.n_users = user_cnt

for split in range(0, splits):
    dir_name = args.out+'_'+str(split)
    dir_path = os.path.join(data_path, dir_name)
    try:
        if not os.path.isfile(dir_path):
            os.mkdir(dir_path)
            os.mkdir(os.path.join(dir_path, TRAIN))
            os.mkdir(os.path.join(dir_path, EVAL))
        args.s = seed_list[split]
        args.i = split
        preprocess(args)

        data = {
            'val_percentage' : '20%',
            'test_percentage': '20%',
            'Random seed': args.s
        }
        with open(os.path.join(dir_path, 'config.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except OSError:
        print("Failed run on creating split %s" % dir_name)
    else:
        print("Successfully created split %s " % dir_name)

for neighbour in neighbours:
    for split in range(0, splits):
        n_dir_name = str(neighbour).zfill(3) + '_' + str(split)
        n_dir_path = os.path.join(data_path, n_dir_name)
        try:
            if not os.path.isfile(n_dir_path):
                os.mkdir(n_dir_path)
                os.mkdir(os.path.join(n_dir_path, TRAIN))
                os.mkdir(os.path.join(n_dir_path, EVAL))
        except OSError:
            print("Failed run on creating split %s" % n_dir_name)
        else:
            print("Successfully created split %s " % n_dir_name)

for neighbour in neighbours:
    args.n = neighbour
    generate_similar_pair(args)
print(" Similar user and simlar items pairs generated.")

for neighbour in neighbours:
    for split in range(0, splits):
        for evalType in eval_types:
            args.n = neighbour
            args.type = evalType
            args.i = split
            generate_user_item_pair(args)
            #generate_blocking_target(args)

for neighbour in neighbours:
    for split in range(0, splits):
        for evalType in eval_types:
            args.n = neighbour
            args.type = evalType
            args.i = split
            get_baseline_output(args)
            print('Baseline output for split', (neighbour, split))

for split in range(0, splits):
    # load user ratings
    for evalType in eval_types:
        args.i = split
        args.type = evalType
        args.s = seed_list[split]
        run_baseline(args)
        print("Ran baseline on split %d" % split)

'''

print("Creating data for PSL.")
for neighbour in neighbours:
    for split in range(0, splits):
        dir_name = args.out+'_'+str(split)
        dir_path = os.path.join(data_path, dir_name)
        n_dir_name = str(neighbour).zfill(3)+'_'+str(split)
        n_dir_path = os.path.join(data_path, n_dir_name)
        try:
            if not os.path.isfile(n_dir_path):
                os.mkdir(n_dir_path)
            args.i = split
            args.n = neighbour
            args.s = seed_list[split]
            generate_user_item_pair(args)
            generate_blocking_target(args)
            data = {
                'test_percentage': '20%',
                'neighbour': args.n,
                'Random seed': args.s
            }
            with open(os.path.join(n_dir_path, 'config.json'), 'w') as outfile:
                json.dump(data, outfile, indent=4)

        except OSError:
            print("Failed run on creating split %s ." % n_dir_name)
        else:
            print("Successfully created split %s ." % n_dir_name)


for split in range(0, splits):
    args.i = split
    print("Starting baseline on split %s" % str(split))
    args.s = seed_list[split]
    args.list = neighbours
    startBaseline(args)
    print("Finished on split %s" % str(split))
'''

