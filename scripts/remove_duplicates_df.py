# check duplicates and write into file

import pandas as pd
import os

def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df

Dataset = 'movie'
INPUT_FILE = 'generated_data.txt'
OUTPUT_FILE = 'generated_data_noduplicates.txt'

target_inputpath = os.path.join('..', 'data', Dataset, INPUT_FILE)

target_outputpath = os.path.join('..', 'data', Dataset, OUTPUT_FILE)

target_df = readfile(target_inputpath)

print(' before', target_df.shape)

target_df = df.drop_duplicates()

print(' after', target_df.shape)

row, col = target_df.shape
targetmatrix = target_df.values

with open(target_outputpath, 'w') as targetWriter:
    for i in range(0, row):
        user = targetmatrix[i][0]
        item = targetmatrix[i][1]

        targetWriter.write('%d\t%d\n' % (user, item))