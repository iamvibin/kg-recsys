import pandas as pd
import numpy as np
import os
import argparse
from scipy.sparse.linalg import svds


def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df

Dataset = 'movie'
neighbours = 40 #20 #50
INPUT_FILE = 'ratings_scores_obs.txt'
INPUT_KG_FILE = 'relations_obs.txt'
SIM_USER_OUTPUT_FILE = 'sim_users_50_obs.txt'
SIM_ITEM_OUTPUT_FILE = 'sim_items_50_obs.txt'

OBS_FILE = 'ratings_obs.txt'
ALL_DATA_FILE = 'ratings_scores.txt'
NEW_INTERACTIONS_FILE = 'generated_relations_from_kg.txt'

obs_inputpath = os.path.join('..', 'data', Dataset, OBS_FILE)
kg_inputpath = os.path.join('..', 'data', Dataset, INPUT_KG_FILE)
full_inputpath = os.path.join('..','data', Dataset, ALL_DATA_FILE)
outputpath = os.path.join('..', 'data', Dataset, NEW_INTERACTIONS_FILE)

inputpath = os.path.join('..', 'data', Dataset, INPUT_FILE)
df = readfile(inputpath)

R_df = df.pivot(index='users', columns='items', values='ratings').fillna(0)

R = R_df.values
user_ratings_mean = np.mean(R, axis=1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(R_demeaned, k=neighbours)

user_sim_matrix = np.dot(U, np.transpose(U))
item_sim_matrix = np.dot(np.transpose(Vt), Vt)

idx = -neighbours-1
similar_users = np.argpartition(user_sim_matrix, np.argmax(user_sim_matrix, axis=0))[:, idx:]
similar_items = np.argpartition(item_sim_matrix, np.argmax(item_sim_matrix, axis=0))[:, idx:]
item_limit , it = similar_items.shape

threshold = 4
obs_df = readfile(obs_inputpath)
full_df = readfile(full_inputpath)
matrix = obs_df.values
full_matrix = full_df.values
# storing the known user item interactions
dictionary = {}

row, col = full_matrix.shape
for index in range(0, row):
    user = full_matrix[index][0]
    item = full_matrix[index][1]
    rating = full_matrix[index][2]
    if rating >= threshold:
        rating = 1
    else:
        rating = 0
    if user not in dictionary:
        dictionary[int(user)] = {}
    dictionary[int(user)][int(item)] = int(rating)

kg_df = pd.read_csv(kg_inputpath, delimiter="\t", header=None, names=["item1", "relation", "item2"])
kg_matrix = kg_df.values
row, col = kg_matrix.shape
kg_dict = {}
for index in range(0, row):
    item1 = int(kg_matrix[index][0])
    item2 = int(kg_matrix[index][2])
    if item1 not in kg_dict:
        kg_dict[item1] = set()
    kg_dict[item1].add(item2)
print(len(kg_dict))

# getting new combinations
row, col = matrix.shape
with open(outputpath, 'w') as writer:
    for index in range(0, row):
        user = matrix[index][0]
        item = matrix[index][1]
        rating = matrix[index][2]

        if rating > 0:
            userlist = similar_users[int(user)][:]
            userlist = userlist.reshape(-1, 1)
            if item in kg_dict:
                list_of_items = list(kg_dict[item])
                for each_sample_item in list_of_items:
                    if each_sample_item < item_limit:
                        itemlist = similar_items[int(each_sample_item)][:]
                    else:
                        itemlist = np.asarray(each_sample_item)

                    itemlist = itemlist.reshape(-1, 1)
                    possiblepairs = np.dstack(np.meshgrid(userlist, itemlist)).reshape(-1, 2)
                    r, c = possiblepairs.shape

                    for i in range(0, r):
                        newuser = possiblepairs[i][0]
                        newmovie = possiblepairs[i][1]

                        if newuser in dictionary:
                            existingList = dictionary[newuser]
                            if newmovie in existingList:
                                continue
                        else:
                            dictionary[newuser] = {}
                        writer.write('%d\t%d\n' % (newuser, newmovie))
                        dictionary[newuser][newmovie] = 0
