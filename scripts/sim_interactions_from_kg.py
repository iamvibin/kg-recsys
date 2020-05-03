import pandas as pd
import numpy as np
import os
import argparse
from scipy.sparse.linalg import svds


def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df


def generate_user_item_pair(args):

    Dataset = args.d
    neighbour = args.n
    threshold = args.t
    split = args.i
    dir_name = args.out+'_'+str(split)
    n_dir_name = str(neighbour).zfill(3)+'_'+str(split)

    ALL_RATINGS_INPUT_FILE = 'ratings_scores.txt'
    INPUT_KG_FILE = 'relations_obs.txt'
    OBS_FILE = 'ratings_obs.txt'
    SIM_USER_OUTPUT_FILE = 'sim_users_obs.txt'
    SIM_ITEM_OUTPUT_FILE = 'sim_items_obs.txt'

    IMPLICIT_INTERACTIONS_FILE = 'generated_relations_from_ratings.txt'
    NEW_INTERACTIONS_FILE = 'generated_relations_from_kg.txt'

    # paths
    ratings_inputpath = os.path.join('..', 'data', Dataset, dir_name, ALL_RATINGS_INPUT_FILE)
    obs_inputpath = os.path.join('..', 'data', Dataset, dir_name, OBS_FILE)
    kg_inputpath = os.path.join('..', 'data', Dataset, dir_name, INPUT_KG_FILE)
    full_inputpath = os.path.join('..', 'data', Dataset, dir_name, ALL_RATINGS_INPUT_FILE)
    implicit_sim_pair_file_path = os.path.join('..', 'data', Dataset, n_dir_name, IMPLICIT_INTERACTIONS_FILE)
    outputpath = os.path.join('..', 'data', Dataset, n_dir_name, NEW_INTERACTIONS_FILE)
    uu_outputpath = os.path.join('..', 'data', Dataset, n_dir_name, SIM_USER_OUTPUT_FILE)
    ii_outputpath = os.path.join('..', 'data', Dataset, n_dir_name, SIM_ITEM_OUTPUT_FILE)

    print("Calculating similar users and similar items from implicit ratings")
    df = readfile(ratings_inputpath)

    R_df = df.pivot(index='users', columns='items', values='ratings').fillna(0)

    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)

    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=neighbour)

    user_sim_matrix = np.dot(U, np.transpose(U))
    item_sim_matrix = np.dot(np.transpose(Vt), Vt)

    idx = -neighbour - 1
    similar_users = np.argpartition(user_sim_matrix, np.argmax(user_sim_matrix, axis=0))[:, idx:]
    similar_items = np.argpartition(item_sim_matrix, np.argmax(item_sim_matrix, axis=0))[:, idx:]
    item_limit, it = similar_items.shape

    row, col = similar_users.shape
    with open(uu_outputpath, 'w') as sim_user_writer:
        for user in range(0, row):
            for c_index in range(0, col):
                if similar_users[user][c_index] == user:
                    continue
                else:
                    sim_user_writer.write('%d\t%d\t1\n' % (user, similar_users[user][c_index]))
    sim_user_writer.close()

    row, col = similar_items.shape
    with open(ii_outputpath, 'w') as sim_item_writer:
        for item in range(0, row):
            for c_index in range(0, col):
                if similar_items[item][c_index] == item:
                    continue
                else:
                    sim_item_writer.write('%d\t%d\t1\n' % (item, similar_items[item][c_index]))
    sim_item_writer.close()

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

    # Storing all the relations from KG in dictionary
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
    print("Number of entities", len(kg_dict))

    # getting new combinations from implicit ratings
    print("Writing new interactions from implicit ratings")

    row, col = matrix.shape
    with open(implicit_sim_pair_file_path, 'w') as writer:
        for index in range(0, row):
            user = int(matrix[index][0])
            item = int(matrix[index][1])
            rating = float(matrix[index][2])

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

    desired_obs = obs_df[obs_df['ratings'] > 0]
    obs_without_ratings_df = desired_obs.drop('ratings', 1)
    new_potential_pairs_df = pd.read_csv(implicit_sim_pair_file_path, delimiter="\t", header=None,
                                         names=["users", "items"])

    all_pairs_df = pd.concat([obs_without_ratings_df, new_potential_pairs_df])
    all_pairs = all_pairs_df.values
    print("Writing new user item pair interactions from knowledge graph.")

    row, col = all_pairs.shape
    with open(outputpath, 'w') as writer:
        for index in range(0, row):
            user = all_pairs[index][0]
            item = all_pairs[index][1]

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