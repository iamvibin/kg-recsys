import os
import argparse
import pandas as pd
import numpy as np
import operator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

DATASET = 'movie'

TRUTH_FILE = 'ratings_truth.txt'
PREDICTIONS_FILE = 'RATING.txt'
BASELINE_PREDICTION_FILE = 'scores.txt'
RESULT = 'results.txt'
k_set = [1, 2, 5, 10, 20, 50, 100]
threshold = 0.8
splits = 2
neighbours = [2, 5, 10]

def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["user", "item", "rating"])
    return df

def getPrecisionRecallTopK(truth_dict, target_dict, k_set):
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
        f1 = 2 / (1 / precision_list[i] + 1 / recall_list[i]) if precision_list[i] != 0 or recall_list[i]!=0 else 1
        f1_score_list.append(f1)

    return precision_list, recall_list, f1_score_list

def getDict(truth_matrix):
    truth_dict = {}
    r, c = truth_matrix.shape

    for i in range(0, r):
        user = truth_matrix[i][0]
        item = truth_matrix[i][1]
        ratings = truth_matrix[i][2]

        if ratings >= 0:
            if user not in truth_dict:
                truth_dict[user] = set()
            truth_dict[user].add(tuple([item, ratings]))
    return truth_dict


with open(RESULT, 'w') as writer:
    for split in range(0, splits):
        result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

        truth_folder_name = '000' + '_' + str(split)
        baseline_truth_path = os.path.join('..', 'data', DATASET, truth_folder_name, TRUTH_FILE)

        baseline_truth_df = readfile(baseline_truth_path)
        baseline_truth_matrix = baseline_truth_df.values
        baseline_truth_dict = getDict(baseline_truth_matrix)

        baseline_scores_path = os.path.join('..', 'data', DATASET, truth_folder_name, BASELINE_PREDICTION_FILE)
        baseline_target_scores_df = pd.read_csv(baseline_scores_path, delimiter="\t", header=None, names=["prediction"])
        baseline_target_df = pd.concat([baseline_truth_df, baseline_target_scores_df], axis=1).drop('rating', axis=1)

        baseline_target_matrix = baseline_target_df.values
        baseline_target_dict = getDict(baseline_target_matrix)

        baseline_truth_dict = {}
        r, c = baseline_truth_matrix.shape

        for i in range(0, r):
            user = baseline_truth_matrix[i][0]
            item = baseline_truth_matrix[i][1]
            ratings = baseline_truth_matrix[i][2]

            if ratings >= 0:
                if user not in baseline_truth_dict:
                    baseline_truth_dict[user] = set()
                baseline_truth_dict[user].add(item)

        baseline_target_dict = {}
        r, c = baseline_target_matrix.shape

        for i in range(0, r):
            user = baseline_target_matrix[i][0]
            item = baseline_target_matrix[i][1]
            ratings = baseline_target_matrix[i][2]

            if ratings >= 0:
                if user not in baseline_target_dict:
                    baseline_target_dict[user] = set()
                baseline_target_dict[user].add(tuple([item, ratings]))

        baseline_precision_list, baseline_recall_list, baseline_f1_score_list = getPrecisionRecallTopK(
            baseline_truth_dict, baseline_target_dict, k_set)

        new_df = pd.merge(baseline_truth_df, baseline_target_df, how='left', left_on=['user', 'item'],
                          right_on=['user', 'item'])
        labels = new_df['rating'].to_list()
        scores = new_df['prediction'].to_list()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        print('auc is', auc)
        auprc = average_precision_score(y_true=labels, y_score=scores)
        print('auprc is', auprc)
        fpr, tpr, _ = roc_curve(labels, scores)
        result_table = result_table.append({'classifiers': truth_folder_name,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

        writer.write('%d\t%d\t%f\t%f\t' % (0, split, auc, auprc))
        for listitem in baseline_precision_list:
            writer.write('%f\t' % (listitem))
        for listitem in baseline_recall_list:
            writer.write('%f\t' % (listitem))
        for listitem in baseline_f1_score_list:
            writer.write('%f\t' % (listitem))
        writer.write('\n')
        print(baseline_precision_list)
        print(baseline_recall_list)
        print(baseline_f1_score_list)

        for neighbour in neighbours:
            folder_name = str(neighbour).zfill(3) + '_' + str(split)
            predictions_path = os.path.join('..', 'cli', 'inferred-predicates', folder_name, PREDICTIONS_FILE)

            truth_df = baseline_truth_df
            target_df = readfile(predictions_path)

            new_df = pd.merge(truth_df, target_df, how='left', left_on=['user', 'item'], right_on=['user', 'item'])
            labels = new_df['rating_x'].to_list()
            scores = new_df['rating_y'].to_list()
            auc = roc_auc_score(y_true=labels, y_score=scores)
            print('auc is', auc)
            auprc = average_precision_score(y_true=labels, y_score=scores)
            print('auprc is', auprc)
            fpr, tpr, _ = roc_curve(labels, scores)
            result_table = result_table.append({'classifiers': folder_name,
                                                'fpr': fpr,
                                                'tpr': tpr,
                                                'auc': auc}, ignore_index=True)

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

            precision_list, recall_list, f1_score_list = getPrecisionRecallTopK(truth_dict, target_dict, k_set)

            print(precision_list)
            print(recall_list)
            print(f1_score_list)

            writer.write('%d\t%d\t%f\t%f\t' % (neighbour, split, auc, auprc))
            for listitem in precision_list:
                writer.write('%f\t' % (listitem))
            for listitem in recall_list:
                writer.write('%f\t' % (listitem))
            for listitem in f1_score_list:
                writer.write('%f\t' % (listitem))
            writer.write('\n')

        result_table.set_index('classifiers', inplace=True)
        fig = plt.figure(figsize=(8, 6))

        for i in result_table.index:
            plt.plot(result_table.loc[i]['fpr'],
                     result_table.loc[i]['tpr'],
                     label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Flase Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 13}, loc='lower right')

        fig_name = truth_folder_name + '.png'
        fig.savefig(fig_name)
        print(truth_folder_name + 'done')