import numpy as np
import pandas as pd
import os
from BPMF.recommend.bpmf import BPMF
from BPMF.recommend.utils.evaluation import RMSE

SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})


def file_writer(path, target_matrix, preds):
    writer = open(path, 'w')
    size = len(target_matrix)

    for i in range(0, size):
        user = target_matrix[i][0]
        item = target_matrix[i][1]
        rating = preds[i]
        writer.write('%d\t%d\t%f\n' % (user, item, rating))

    writer.close()

def get_baseline_output(args):
    split = args.i
    neighbour = args.n
    DIR = args.out + '_' + str(split)
    evalType = args.type

    INPUT_FILE = 'ratings_obs.txt'
    TEST_FILE = 'ratings_target'
    OUTPUT_FILE = 'baseline_output'
    data_path = os.path.join('..', 'data', args.d)
    files = ['_without_kg.txt', '_with_kg.txt']

    INPUT_PATH = os.path.join(data_path, DIR, evalType, INPUT_FILE)

    df = pd.read_csv(INPUT_PATH, delimiter='\t', header=None, names=["users", "items", "ratings"])
    df['ratings'] = df['ratings'] * 5
    df = df[list(df.columns)].astype(int)
    ratings = df.values

    n_user = args.n_users#max(ratings[:, 0]) + 1
    n_item = args.n_items

    # fit model
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=4,
                max_rating=5., min_rating=1., seed=args.s).fit(ratings, n_iters=10)

    for file_type in files:
        n_dir_name = str(neighbour).zfill(3) + '_' + str(split)
        test_file_name = TEST_FILE + file_type
        output_file_name = OUTPUT_FILE + file_type
        TARGET_PATH = os.path.join(data_path, n_dir_name, evalType, test_file_name)
        BASELINE_OUTPUT_PATH = os.path.join(data_path, n_dir_name, evalType, output_file_name)

        df = pd.read_csv(TARGET_PATH, delimiter="\t", header=None, names=["users", "items"])
        df = df[list(df.columns)].astype(int)
        target_matrix = df.values

        preds = bpmf.predict(target_matrix)
        preds = preds / 5

        file_writer(BASELINE_OUTPUT_PATH, target_matrix, preds)

def run_baseline(args):
    split = args.i
    DIR = args.out + '_' + str(split)
    eval = args.type
    INPUT_FILE = 'ratings_obs.txt'
    TEST_FILE = 'ratings_target.txt'
    OUTPUT_FILE = 'baseline_output.txt'
    data_path = os.path.join('..', 'data', args.d)

    INPUT_PATH = os.path.join(data_path, DIR, eval, INPUT_FILE)
    TARGET_PATH = os.path.join(data_path, DIR, eval, TEST_FILE)
    BASELINE_OUTPUT_PATH = os.path.join(data_path, DIR, eval, OUTPUT_FILE)

    df = pd.read_csv(INPUT_PATH, delimiter='\t', header=None, names=["users", "items", "ratings"])
    df['ratings'] = df['ratings'] * 5
    df = df[list(df.columns)].astype(int)
    ratings = df.values

    #n_user = max(ratings[:, 0])+1
    #n_item = max(ratings[:, 1])+1
    n_user = args.n_users  # max(ratings[:, 0]) + 1
    n_item = args.n_items

    # fit model
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=4,
                max_rating=5., min_rating=1., seed=args.s).fit(ratings, n_iters=10)
    RMSE(bpmf.predict(ratings[:, :2]), ratings[:, 2])  # training RMSE

    df = pd.read_csv(TARGET_PATH, delimiter="\t", header=None, names=["users", "items"])
    df = df[list(df.columns)].astype(int)
    target_matrix = df.values

    preds = bpmf.predict(target_matrix)
    preds = preds / 5

    file_writer(BASELINE_OUTPUT_PATH, target_matrix, preds)

