import os
import argparse
import pandas as pd
import operator
from sklearn.metrics import roc_auc_score


def readfile(path):
    df = pd.read_csv(path, delimiter = "\t", header=None, names=["user", "item", "rating"])
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='movie', help='which dataset to use')
args = parser.parse_args()
DATASET = args.d

TRUTH_FILE = 'ratings_truth.txt'
PREDICTIONS_FILE = 'RATING.txt'
RESULT='results.txt'
k_set = [1, 2, 5, 10, 20, 50, 100]
threshold = 0.8
splits = 10
neighbours = [5, 10]

with open(RESULT,'w') as writer:
    for neighbour in neighbours:
        for split in range(0, splits):
            folder_name = str(neighbour) + '_' + str(split)

            truth_path = os.path.join('..', 'data', DATASET, folder_name, TRUTH_FILE)
            predictions_path = os.path.join('..', 'cli', 'inferred-predicates', folder_name, PREDICTIONS_FILE)

            truth_df = readfile(truth_path)
            target_df = readfile(predictions_path)

            new_df = pd.merge(truth_df, target_df, how='left', left_on=['user', 'item'], right_on=['user', 'item'])
            labels = new_df['rating_x'].to_list()
            scores = new_df['rating_y'].to_list()
            auc = roc_auc_score(y_true=labels, y_score=scores)
            print('auc is', auc)

            # In[12]:

            truth_matrix = truth_df.values
            target_matrix = target_df.values

            # In[13]:

            truth_dict = {}
            r, c = truth_matrix.shape

            for i in range(0, r):
                user = truth_matrix[i][0]
                item = truth_matrix[i][1]
                ratings = truth_matrix[i][2]

                if ratings >= threshold:
                    if user not in truth_dict:
                        truth_dict[user] = set()
                    truth_dict[user].add(item)

            target_dict = {}
            r, c = target_matrix.shape

            for i in range(0, r):
                user = target_matrix[i][0]
                item = target_matrix[i][1]
                ratings = target_matrix[i][2]

                if ratings >= threshold:
                    if user not in target_dict:
                        target_dict[user] = set()
                    target_dict[user].add(tuple([item, ratings]))

                    # In[14]:

            precision_list = []
            recall_list = []
            f1_score_list = []
            user_count = 0
            for k in k_set:
                item_count = 0
                relevant_item_count = 0
                recommended_item_count = 0
                for user, truth_set in truth_dict.items():
                    if user not in target_dict:
                        continue
                    prediction_set = target_dict[user]
                    prediction_list = list(prediction_set)
                    prediction_list.sort(key=operator.itemgetter(1), reverse=True)

                    for i in range(0, min(k, len(prediction_list))):
                        item = prediction_list[i][0]
                        if item in truth_set:
                            item_count += 1

                    relevant_item_count += len(truth_set)
                    recommended_item_count += len(prediction_set)

                    precision = item_count / recommended_item_count
                    recall = item_count / relevant_item_count

                precision_list.append(precision)
                recall_list.append(recall)

            for i in range(len(k_set)):
                f1 = 2 / (1 / precision_list[i] + 1 / recall_list[i])
                f1_score_list.append(f1)

            print(precision_list)
            print(recall_list)
            print(f1_score_list)

            writer.write('%d\t%d\t%f\t' % (neighbour, split, auc))
            for listitem in precision_list:
                writer.write('%f\t' % (listitem))
            for listitem in recall_list:
                writer.write('%f\t' % (listitem))
            for listitem in f1_score_list:
                writer.write('%f\t' % (listitem))
            writer.write('\n')
