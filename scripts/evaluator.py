import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import pandas as pd
from collections import OrderedDict
import argparse


def getLabelsPredictions(fr_truth, fr_prediction):
    truthDict = {}
    for line in fr_truth:
        array = line.split('\t')
        user = array[0]
        item = array[1]
        rating = array[2].replace('\n', '')
        truthDict[user + '\t' + item] = rating

    labels = []
    scores = []
    for line in fr_prediction:
        array = line.split('\t')
        user = array[0].replace("'", '')
        item = array[1].replace("'", '')
        rating = array[2].replace('\n', '')

        key = user + '\t' + item
        if key in truthDict:
            labels.append(float(truthDict[key]))
            scores.append(float(rating))

    return labels, scores


def getPrecisionAtK(fr_truth, fr_prediction, k):
    truePredictions = {}

    for line in fr_truth:
        array = line.split('\t')
        user = array[0]
        item = array[1]
        rating = float(array[2].replace('\n', ''))
        if rating > 0:
            if user not in truePredictions:
                itemSet = set()
                itemSet.add(item)
                truePredictions[user] = itemSet
            else:
                truePredictions[user].add(item)

    output = {}

    for line in fr_prediction:
        array = line.split('\t')
        user = array[0].replace("'", '')
        item = array[1].replace("'", '')
        rating = float(array[2].replace('\n', ''))

        if rating > 0:
            if user not in output:
                itemSet = set()
                itemSet.add(item)
                output[user] = itemSet
            else:
                output[user].add(item)

    precision = 0

    for user in output:
        predicted_set = output[user]
        true_set = truePredictions[user]
        l = min(k, len(predicted_set))
        predicted_list = list(predicted_set)
        for i in range(0, l):
            item = predicted_list[i]
            if item in true_set:
                precision = precision + 1
    result = precision / k
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''' Get ROC_AUC  Scores''')

    parser.add_argument('--truth', type=str, dest='truth_file', default='../data/movie/ratings_truth.txt')
    parser.add_argument('--predictions', type=str, dest='prediction_file', default='../data/movie/RATING.txt')

    parsed_args = parser.parse_args()

    prediction_file = parsed_args.prediction_file
    truth_file = parsed_args.truth_file

    fr_prediction = open(prediction_file, 'r')
    fr_truth = open(truth_file, 'r')

    labels, scores = getLabelsPredictions(fr_truth, fr_prediction)
    
    category = [1 if i >= 0.8 else 0 for i in labels]

    predictions = [1 if i >= 0.5 else 0 for i in scores]

    precision = precision_score(category, predictions, average='macro')
    auc = roc_auc_score(y_true=category, y_score=predictions)

    print(auc)
    print(precision)

    precisionAtK = getPrecisionAtK(fr_truth, fr_prediction, 2)
    print(precisionAtK)
    fr_truth.close()
    fr_prediction.close()
