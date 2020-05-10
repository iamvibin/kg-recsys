import pandas as pd
import numpy as np
import os
import argparse
from scipy.sparse.linalg import svds


def removeDuplicatesAndWrite(path, df):
    df.drop_duplicates(keep='first', inplace=True)
    print("len after ", len(df))
    df_matrix = df.values
    r, c = df_matrix.shape
    with open(path, 'w') as Writer:
        for i in range(0, r):
            user = df_matrix[i][0]
            item = df_matrix[i][1]
            Writer.write('%d\t%d\n' % (user, item))

def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df


def generate_user_item_pair(args):

    Dataset = args.d
    neighbour = args.n
    threshold = args.t
    evalType = args.type
    split = args.i
    dir_name = args.out+'_'+str(split)
    n_dir_name = str(neighbour).zfill(3)+'_'+str(split)

    ALL_RATINGS_INPUT_FILE = 'ratings.txt'
    INPUT_KG_FILE = 'relations_obs.txt'
    OBS_FILE = 'ratings_obs.txt'

    IMPLICIT_INTERACTIONS_FILE = 'generated_relations_from_ratings.txt'
    NEW_INTERACTIONS_FILE = 'generated_relations_from_kg.txt'

    # paths
    ratings_inputpath = os.path.join('..', 'data', Dataset, ALL_RATINGS_INPUT_FILE)
    obs_inputpath = os.path.join('..', 'data', Dataset, dir_name, evalType, OBS_FILE)
    kg_inputpath = os.path.join('..', 'data', Dataset, INPUT_KG_FILE)
    full_inputpath = os.path.join('..', 'data', Dataset,  ALL_RATINGS_INPUT_FILE)

    implicit_sim_pair_file_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, IMPLICIT_INTERACTIONS_FILE)
    outputpath = os.path.join('..', 'data', Dataset, n_dir_name, evalType, NEW_INTERACTIONS_FILE)

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
    print("Writing new interactions from implicit ratings for split ", (n_dir_name,evalType))
    new_users = []
    new_items = []

    row, col = matrix.shape
    for index in range(0, row):
        user = int(matrix[index][0])
        item = int(matrix[index][1])
        rating = float(matrix[index][2])

        if rating >= threshold:
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
                # writer.write('%d\t%d\n' % (newuser, newmovie))
                new_users.append(newuser)
                new_items.append(newmovie)
                dictionary[newuser][newmovie] = 0


    desired_obs = obs_df[obs_df['ratings'] > 0]
    obs_without_ratings_df = desired_obs.drop('ratings', 1)
    #new_potential_pairs_df = pd.read_csv(implicit_sim_pair_file_path, delimiter="\t", header=None, names=["users", "items"])

    collected_data = {'users': new_users, 'items': new_items}
    new_potential_pairs_df = pd.DataFrame(collected_data)
    print('new_potential_pairs_df', len(new_potential_pairs_df))

    all_pairs_df = pd.concat([obs_without_ratings_df, new_potential_pairs_df])
    all_pairs = all_pairs_df.values
    print("Writing new user item pair interactions from knowledge graph for split: ", (n_dir_name,evalType))

    row, col = all_pairs.shape
    new_users = []
    new_items = []
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
                    # writer.write('%d\t%d\n' % (newuser, newmovie))
                    new_users.append(newuser)
                    new_items.append(newmovie)
                    dictionary[newuser][newmovie] = 0


    collected_data = {'users': new_users, 'items': new_items}
    new_potential_pairs_df_with_kg = pd.DataFrame(collected_data)
    print('new_potential_pairs_df_with_kg', len(new_potential_pairs_df_with_kg))

    Dataset = args.d
    neighbour = args.n
    threshold = args.t
    evalType = args.type
    split = args.i
    dir_name = args.out + '_' + str(split)
    n_dir_name = str(neighbour).zfill(3) + '_' + str(split)

    IMPLICIT_INTERACTIONS_FILE = 'generated_relations_from_ratings.txt'
    NEW_INTERACTIONS_FILE = 'generated_relations_from_kg.txt'
    TARGET_FILE = 'ratings_target.txt'
    OBS_FILE = 'ratings_obs.txt'
    BLOCKING_WITHOUT_KG_FILE = 'blocking_obs_without_kg.txt'
    BLOCKING_WITH_KG_FILE = 'blocking_obs_with_kg.txt'
    TARGET_WITHOUT_KG_FILE = 'ratings_target_without_kg.txt'
    TARGET_WITH_KG_FILE = 'ratings_target_with_kg.txt'

    obs_filepath = os.path.join('..', 'data', Dataset, dir_name, evalType, OBS_FILE)
    pairs_from_rating_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, IMPLICIT_INTERACTIONS_FILE)
    pairs_from_KG_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, NEW_INTERACTIONS_FILE)
    target_input_path = os.path.join('..', 'data', Dataset, dir_name, evalType, TARGET_FILE)
    target_without_kg_output_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, TARGET_WITHOUT_KG_FILE)
    target_with_kg_output_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, TARGET_WITH_KG_FILE)
    blocking_without_kg_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, BLOCKING_WITHOUT_KG_FILE)
    blocking_with_kg_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, BLOCKING_WITH_KG_FILE)

    obs_df_ratings = pd.read_csv(obs_filepath, delimiter="\t", header=None, names=["users", "items", "ratings"])
    obs_df = obs_df_ratings.drop('ratings', 1)
    target_df = pd.read_csv(target_input_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_ratings_df = new_potential_pairs_df #pd.read_csv(pairs_from_rating_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_KG_df = new_potential_pairs_df_with_kg #pd.read_csv(pairs_from_KG_path, delimiter="\t", header=None, names=["users", "items"])

    blocking_df_without_kg = pd.concat([obs_df, target_df, target_from_ratings_df])
    print("len before blocking_df_without_kg ", len(blocking_df_without_kg))
    removeDuplicatesAndWrite(blocking_without_kg_path, blocking_df_without_kg)

    blocking_df_with_kg = pd.concat([obs_df, target_df, target_from_KG_df, target_from_ratings_df])
    print("blocking_df_with_kg ", len(blocking_df_with_kg))
    removeDuplicatesAndWrite(blocking_with_kg_path, blocking_df_with_kg)

    net_target_df_without_kg = pd.concat([target_df, target_from_ratings_df])
    print(" len before net_target_df_without_kg ", len(net_target_df_without_kg))
    removeDuplicatesAndWrite(target_without_kg_output_path, net_target_df_without_kg)

    net_target_df_with_kg = pd.concat([target_df, target_from_KG_df, target_from_ratings_df])
    print(" len before net_target_df_with_kg ", len(net_target_df_with_kg))
    removeDuplicatesAndWrite(target_with_kg_output_path, net_target_df_with_kg)

    print("Generated blocking and targets for split", (n_dir_name, evalType))

