import os
import argparse
import shutil
from main import start_datageneration
from src.main import startBaseline
import json
import random
import sys


def copy_files(arr, src, dest):
    """
    Copy each file from src dir to dest dir.
    """
    for item in arr:
        file_path = os.path.join(src, item)

        # if item is a file, copy it
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest)


parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n', type=int, default=5, help='the number of epochs')
parser.add_argument('--t', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--s', type=int, default=555, help='seed value')
args = parser.parse_args()

DATASET = args.d
splits = 1
neighbours = [5, 10, 20]
count = splits * len(neighbours)
seed_list = random.sample(range(0, 2**32-1), count)
iter = 0

path = os.getcwd()
data_path = os.path.join(path, '..', 'data', DATASET)
filenames = os.listdir(data_path)

for split in range(0, splits):
    for neighbour in neighbours:
        args.s = seed_list[iter]
        iter += 1
        folder_name = str(neighbour) + '_' + str(split)
        new_path = os.path.join(path, '..', 'data', DATASET, folder_name)
        args.n = neighbour

        try:
            if not os.path.isfile(path):
                os.mkdir(new_path)
            copy_files(filenames, data_path, new_path)
            args.d = os.path.join(DATASET, folder_name)
            start_datageneration(args)
            startBaseline(args)

            data = {}
            data[folder_name] = []
            data[folder_name].append({
                'test_percentage': '20%',
                'Number of neighbours': neighbour,
                'Random seed': args.s
            })
            with open(os.path.join(new_path, 'config.json'), 'w') as outfile:
                json.dump(data, outfile)
        except OSError:
            print("Failed run on split" % folder_name)
        else:
            print("Successfully ran split %s " % folder_name)
