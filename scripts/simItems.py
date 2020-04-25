import numpy as np
import pandas as pd
import os
from scipy.sparse.linalg import svds


def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df


Dataset = 'movie'
INPUT_FILE = 'ratings_scores_obs.txt'
SIM_USER_OUTPUT_FILE = 'sim_users_50_obs.txt'
SIM_ITEM_OUTPUT_FILE = 'sim_items_50_obs.txt'
nearest_neighbour = 20
inputpath = os.path.join('..', 'data', Dataset, INPUT_FILE)
df = readfile(inputpath)

R_df = df.pivot(index='users', columns='items', values='ratings').fillna(0)

R = R_df.values
user_ratings_mean = np.mean(R, axis=1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(R_demeaned, k=nearest_neighbour)

user_sim_matrix = np.dot(U, np.transpose(U))
item_sim_matrix = np.dot(np.transpose(Vt), Vt)

idx = -nearest_neighbour - 1
similar_users = np.argpartition(user_sim_matrix, np.argmin(user_sim_matrix, axis=0))[:, idx:]
similar_items = np.argpartition(item_sim_matrix, np.argmin(item_sim_matrix, axis=0))[:, idx:]

outputpath = os.path.join('..', 'data', Dataset, SIM_USER_OUTPUT_FILE)
row, col = similar_users.shape
with open(outputpath, 'w') as sim_user_writer:
    for user in range(0, row):
        for c_index in range(0, col):
            if similar_users[user][c_index] == user:
                continue
            else:
                sim_user_writer.write('%d\t%d\t1\n' % (user, similar_users[user][c_index]))

outputpath = os.path.join('..', 'data', Dataset, SIM_ITEM_OUTPUT_FILE)
row, col = similar_items.shape
with open(outputpath, 'w') as sim_item_writer:
    for item in range(0, row):
        for c_index in range(0, col):
            if similar_items[item][c_index] == item:
                continue
            else:
                sim_item_writer.write('%d\t%d\t1\n' % (item, similar_items[item][c_index]))

OBS_FILE = 'ratings_obs.txt'
ALL_DATA_FILE = 'ratings_scores.txt'
NEW_INTERACTIONS_FILE = 'generated_ratings.txt'

obs_inputpath = os.path.join('..', 'data', Dataset, OBS_FILE)
full_inputpath = os.path.join('..','data', Dataset, ALL_DATA_FILE)
outputpath = os.path.join('..', 'data', Dataset, NEW_INTERACTIONS_FILE)

threshold = 4
obs_df = readfile(obs_inputpath)
full_df = readfile(full_inputpath)
matrix = obs_df.values
full_matrix = full_df.values
# storing the known user item interactions
dictionary = {}
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
            itemlist = similar_items[int(item)][:]
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
