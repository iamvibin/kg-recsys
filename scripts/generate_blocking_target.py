import pandas as pd
import os
import argparse

def generate_blocking_target(args):
    Dataset = args.d
    IMPLICIT_INTERACTIONS_FILE = 'generated_relations_from_ratings.txt'
    NEW_INTERACTIONS_FILE = 'generated_relations_from_kg.txt'
    TARGET_FILE = 'ratings_target.txt'
    OBS_FILE = 'ratings_obs.txt'
    TARGET_BACKUP_FILE = 'ratings_target_backup.txt'
    BLOCKING_FILE = 'blocking_obs.txt'

    obs_filepath = os.path.join('..', 'data', Dataset, OBS_FILE)
    pairs_from_rating_path = os.path.join('..', 'data', Dataset, IMPLICIT_INTERACTIONS_FILE)
    pairs_from_KG_path = os.path.join('..', 'data', Dataset, NEW_INTERACTIONS_FILE)
    target_path = os.path.join('..', 'data', Dataset, TARGET_FILE)
    target_backup_path = os.path.join('..', 'data', Dataset, TARGET_BACKUP_FILE)
    blocking_path = os.path.join('..', 'data', Dataset, BLOCKING_FILE)

    obs_df_ratings = pd.read_csv(obs_filepath, delimiter="\t", header=None, names=["users", "items", "ratings"])
    obs_df = obs_df_ratings.drop('ratings', 1)
    target_df = pd.read_csv(target_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_ratings_df = pd.read_csv(pairs_from_rating_path, delimiter="\t", header=None, names=["users", "items"])
    target_from_KG_df = pd.read_csv(pairs_from_KG_path, delimiter="\t", header=None, names=["users", "items"])

    blocking_df = pd.concat([obs_df, target_df, target_from_KG_df, target_from_ratings_df])
    print(len(blocking_df))
    blocking_df.drop_duplicates(keep='first', inplace=True)
    print(len(blocking_df))
    net_target_df = pd.concat([target_df, target_from_KG_df, target_from_ratings_df])
    print(len(net_target_df))
    net_target_df.drop_duplicates(keep='first', inplace=True)
    print(len(net_target_df))

    os.rename(target_path, target_backup_path)

    target_matrix = net_target_df.values
    r, c = target_matrix.shape
    with open(target_path, 'w') as targetWriter:
        for i in range(0, r):
            user = target_matrix[i][0]
            item = target_matrix[i][1]
            targetWriter.write('%d\t%d\n' % (user, item))

    blocking_matrix = blocking_df.values
    r, c = blocking_matrix.shape
    with open(blocking_path, 'w') as blockingWriter:
        for i in range(0, r):
            user = blocking_matrix[i][0]
            item = blocking_matrix[i][1]
            blockingWriter.write('%d\t%d\n' % (user, item))


