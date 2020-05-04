import os
import random
import argparse
import shutil
import json
from preprocess import preprocess
from sim_interactions_from_kg import generate_user_item_pair
from generate_blocking_target import generate_blocking_target
from src.main import startBaseline


parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n', type=int, default=5, help='the number of neighbors')
parser.add_argument('--t', type=int, default=4, help='threshold')
parser.add_argument('--s', type=int, default=555, help='seed value')
parser.add_argument('--i', type=int, default=0, help='running on split')
parser.add_argument('--out', type=str, default='000', help='default directory')
parser.add_argument('--list', type=list, default=[], help='list of neighbours')

args = parser.parse_args()

splits = 10
seed_list = random.sample(range(0, 2**32-1), splits)

path = os.getcwd()
data_path = os.path.join(path, '..', 'data', args.d)

neighbours = [2, 5, 10]
for split in range(0, splits):
    dir_name = args.out+'_'+str(split)
    dir_path = os.path.join(data_path, dir_name)
    try:
        if not os.path.isfile(dir_path):
            os.mkdir(dir_path)
        args.s = seed_list[split]
        args.i = split
        preprocess(args)

        data = {
            'test_percentage': '20%',
            'Random seed': args.s
        }
        with open(os.path.join(dir_path, 'config.json'), 'w') as outfile:
            json.dump(data, outfile)
    except OSError:
        print("Failed run on split %s" % dir_name)
    else:
        print("Successfully ran split %s " % dir_name)

neighbours = [2, 5, 10]

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
                json.dump(data, outfile)

        except OSError:
            print("Failed run on split %s" % dir_name)
        else:
            print("Successfully ran split %s " % dir_name)


for split in range(0, splits):
    args.i = split
    args.s = seed_list[split]
    args.list = neighbours
    startBaseline(args)


