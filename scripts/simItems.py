import numpy as np
import pandas as pd
import os
from scipy.sparse.linalg import svds

def readfile(path):
    df = pd.read_csv(path, delimiter = "\t", header=None, names=["users", "items", "ratings"])
    return df

Dataset = 'movie'
INPUT_FILE = 'ratings_scores.txt'
SIM_USER_OUTPUT_FILE = 'sim_users_50_obs.txt'
SIM_ITEM_OUTPUT_FILE = 'sim_items_50_obs.txt'



inputpath = os.path.join('..','data',Dataset,INPUT_FILE)
df = readfile(inputpath)

R_df = df.pivot(index = 'users', columns ='items', values = 'ratings').fillna(0)

R = R_df.values
user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(R_demeaned, k = 50)

user_sim_matrix = np.dot(U, np.transpose(U))
item_sim_matrix = np.dot(np.transpose(Vt), Vt)

similar_users = np.argpartition(user_sim_matrix, np.argmin(user_sim_matrix, axis=0))[:, -51:]
similar_items = np.argpartition(item_sim_matrix, np.argmin(item_sim_matrix, axis=0))[:, -51:]

outputpath = os.path.join('..','Data',Dataset,SIM_USER_OUTPUT_FILE)
row, col = similar_users.shape
with open(outputpath, 'w') as sim_user_writer:
    for user in range(0,row):
        for c_index in range(0,col):
            if similar_users[user][c_index]==user:
                continue
            else:
                sim_user_writer.write('%d\t%d\t1\n' % (user, similar_users[user][c_index]) )

outputpath = os.path.join('..','Data',Dataset,SIM_ITEM_OUTPUT_FILE)
row, col = similar_items.shape
with open(outputpath, 'w') as sim_item_writer:
    for item in range(0,row):
        for c_index in range(0,col):
            if similar_items[item][c_index]==item:
                continue
            else:
                sim_item_writer.write('%d\t%d\t1\n' % (user, similar_items[item][c_index]) )