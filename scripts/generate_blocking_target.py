import pandas as pd
import os
import argparse


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


def generate_blocking_target(args):

    Dataset = args.d
    neighbour = args.n
    threshold = args.t
    evalType = args.type
    split = args.i
    dir_name = args.out+'_'+str(split)
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
    pairs_from_KG_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType,NEW_INTERACTIONS_FILE)
    target_input_path = os.path.join('..', 'data', Dataset, dir_name, evalType, TARGET_FILE)
    target_without_kg_output_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, TARGET_WITHOUT_KG_FILE)
    target_with_kg_output_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, TARGET_WITH_KG_FILE)
    blocking_without_kg_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, BLOCKING_WITHOUT_KG_FILE)
    blocking_with_kg_path = os.path.join('..', 'data', Dataset, n_dir_name, evalType, BLOCKING_WITH_KG_FILE)

    obs_df_ratings = pd.read_csv(obs_filepath, delimiter="\t", header=None, names=["users", "items", "ratings"])
    obs_df = obs_df_ratings.drop('ratings', 1)
    target_df = pd.read_csv(target_input_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_ratings_df = pd.read_csv(pairs_from_rating_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_KG_df = pd.read_csv(pairs_from_KG_path, delimiter="\t", header=None, names=["users", "items"])

    blocking_df_without_kg = pd.concat([obs_df, target_df, target_from_ratings_df])
    print("len before blocking_df_without_kg ",len(blocking_df_without_kg))
    removeDuplicatesAndWrite(blocking_without_kg_path, blocking_df_without_kg)

    blocking_df_with_kg = pd.concat([obs_df, target_df, target_from_KG_df, target_from_ratings_df])
    print("blocking_df_with_kg ", len(blocking_df_with_kg))
    removeDuplicatesAndWrite(blocking_with_kg_path, blocking_df_with_kg)

    net_target_df_with_kg = pd.concat([target_df, target_from_KG_df, target_from_ratings_df])
    print(" len before net_target_df_with_kg ", len(net_target_df_with_kg))
    removeDuplicatesAndWrite(target_with_kg_output_path, net_target_df_with_kg)

    net_target_df_without_kg = pd.concat([target_df, target_from_KG_df, target_from_ratings_df])
    print(" len before net_target_df_with_kg ",len(net_target_df_without_kg))
    removeDuplicatesAndWrite(target_without_kg_output_path, net_target_df_without_kg)

    print("Generated blocking and targets for split", (n_dir_name,evalType))


