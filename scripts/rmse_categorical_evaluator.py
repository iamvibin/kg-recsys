from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n', type=int, default=5, help='the number of neighbors')
parser.add_argument('--t', type=int, default=4, help='threshold')
parser.add_argument('--s', type=int, default=555, help='seed value')
parser.add_argument('--i', type=int, default=0, help='running on split')
parser.add_argument('--out', type=str, default='000', help='default directory')
parser.add_argument('--list', type=list, default=[], help='list of neighbours')
parser.add_argument('--type', type=str, default='eval', help='list of neighbours')
parser.add_argument('--n_items', type=int, default=0, help='number of entities')
parser.add_argument('--n_users', type=int, default=0, help='number of users')
args = parser.parse_args()

TRUTH_FILE = 'ratings_truth.txt'
PREDICTIONS_FILE = 'baseline_output.txt'
OTHER_FILE = 'RATING.txt'
OUTPUT_FILE = 'result.txt'
EVAL = 'eval'
DATA_PATH = os.path.join('..','data','movie')
TRUTH_PATH = os.path.join(DATA_PATH, TRUTH_FILE)
PREDICTIONS_PATH = os.path.join(DATA_PATH, PREDICTIONS_FILE)

neighbours = [0, 2, 5]
splits = 10

with open (OUTPUT_FILE, 'w') as writer:
    for split in range(0, splits):
        for neighbour in neighbours:

            if neighbour == 0:
                DIR_NAME = args.out + '_' + str(split)
                TRUTH_PATH = os.path.join(DATA_PATH, DIR_NAME, EVAL, TRUTH_FILE)
                PREDICTIONS_PATH = os.path.join(DATA_PATH, DIR_NAME, EVAL, PREDICTIONS_FILE)
            else:
                DIR_NAME = str(neighbour).zfill(3) + '_' + str(split)
                TRUTH_PATH = os.path.join(DATA_PATH, args.out + '_' + str(split), EVAL, TRUTH_FILE)
                PREDICTIONS_PATH = os.path.join('..', 'cli', 'inferred-predicates', DIR_NAME, OTHER_FILE)

            truth_df = pd.read_csv(TRUTH_PATH, delimiter="\t", header=None, names=["user", "item", "rating"])
            predictions_df = pd.read_csv(PREDICTIONS_PATH, delimiter="\t", header=None,
                                         names=["user", "item", "prediction"])
            new_df = pd.merge(truth_df, predictions_df, how='left', left_on=['user', 'item'], right_on=['user', 'item'])

            true_ratings = new_df['rating'].to_list()
            predicted_ratings = new_df['prediction'].to_list()

            mse = mean_squared_error(true_ratings, predicted_ratings)
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            rmse = math.sqrt(mse)
            print(rmse, mse, mae)


            writer.write('%d\t%d\t%f\t%f\n' % (neighbour, split, rmse, mae))






