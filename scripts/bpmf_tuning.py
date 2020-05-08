import os
import random
import argparse
import json
import pandas as pd
import operator
from data_reformat import reformat
from preprocess import preprocess
from BPMF.recommend.bpmf import BPMF
from BPMF.recommend.utils.evaluation import RMSE

def run_baseline(args):
    split = args.i
    DIR = args.out + '_' + str(split)
    eval = args.type
    INPUT_FILE = 'ratings_obs.txt'
    TRUTH_FILE = 'ratings_truth.txt'
    data_path = os.path.join('..', 'data', args.d)

    INPUT_PATH = os.path.join(data_path, DIR, eval, INPUT_FILE)
    TRUTH_PATH = os.path.join(data_path, DIR, eval, TRUTH_FILE)

    df = pd.read_csv(INPUT_PATH, delimiter='\t', header=None, names=["users", "items", "ratings"])
    df['ratings'] = df['ratings'] * 5
    df = df[list(df.columns)].astype(int)
    ratings = df.values

    n_user = max(ratings[:, 0])+1
    n_item = max(ratings[:, 1])+1

    # fit model
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=args.dim,
                max_rating=5., min_rating=1., seed=args.s).fit(ratings, n_iters=1000)
    RMSE(bpmf.predict(ratings[:, :2]), ratings[:, 2])  # training RMSE
    print(bpmf.iter_)

    df = pd.read_csv(TRUTH_PATH, delimiter="\t", header=None, names=["users", "items", "ratings"])
    truth = df['ratings']
    truth = truth*5
    truth = truth.astype(int)
    df = df.drop(['ratings'], axis=1)
    df = df[list(df.columns)].astype(int)
    truth_matrix = df.values

    preds = bpmf.predict(truth_matrix)
    rmse = RMSE(preds, truth)
    return rmse


parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n', type=int, default=5, help='the number of neighbors')
parser.add_argument('--t', type=int, default=4, help='threshold')
parser.add_argument('--s', type=int, default=555, help='seed value')
parser.add_argument('--i', type=int, default=0, help='running on split')
parser.add_argument('--out', type=str, default='000', help='default directory')
parser.add_argument('--list', type=list, default=[2], help='list of neighbours')
parser.add_argument('--type', type=str, default='eval', help='list of neighbours')
parser.add_argument('--dim', type=int, default=20, help='number of iterations')

args = parser.parse_args()
EVAL = 'eval'
TRAIN = 'train'
type = [EVAL, TRAIN]
iter_list = [2, 4, 16, 32, 64, 128, 256]
splits = 10
seed_list = random.sample(range(0, 2**32-1), splits)

path = os.getcwd()
data_path = os.path.join(path, '..', 'data', args.d)

entity_id2index ,relation_id2index, item_index_old2new = reformat(args)

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

rmse = {}
for iter in iter_list:
    mean_rmse = 0
    for split in range(0, splits):
        # load user ratings
        args.i = split
        args.s = seed_list[split]
        args.type = EVAL
        args.dim = iter
        rmse = run_baseline(args)
        mean_rmse = mean_rmse + rmse

    mean_rmse=mean_rmse/splits
    print('rmse', rmse)
    if iter not in rmse:
        rmse[iter] = []
    rmse[iter].append(rmse)
print(rmse)
print(max(rmse.iteritems(), key=operator.itemgetter(1))[0])