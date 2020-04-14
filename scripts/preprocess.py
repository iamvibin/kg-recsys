import argparse
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    user_movie_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])

        # Storing user-movie ratings
        if user_index_old not in user_movie_ratings:
            user_movie_ratings[user_index_old] = {}
        user_movie_ratings[user_index_old][item_index] = rating

        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    ratingWriter = open('../data/' + DATASET + '/ratings_scores.txt', 'w', encoding='utf-8')

    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
            itemDict = user_movie_ratings[user_index_old]
            ratingValue = itemDict[item]
            ratingWriter.write('%d\t%d\t%d\n' % (user_index, item, ratingValue))

        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
            ratingValue = 0
            if user_index_old in user_movie_ratings:
                itemDict = user_movie_ratings[user_index_old]
                if item in itemDict:
                    ratingValue = itemDict[item]

            ratingWriter.write('%d\t%d\t%d\n' % (user_index, item, ratingValue))

    writer.close()
    ratingWriter.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    return user_cnt


def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/relations_obs.txt', 'w', encoding='utf-8')
    file = open('../data/' + DATASET + '/kg.txt', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            continue
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))
        writer.write('%d\t%d\t%d\n' % (tail, relation, head))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def data_split():
    print('spliting data into observed and target ...')
    ratio = 0.8

    inputFile = '../data/' + DATASET + '/' + 'ratings_scores.txt'

    observationWriter = open('../data/' + DATASET + '/ratings_obs.txt', 'w', encoding='utf-8')
    targetWriter = open('../data/' + DATASET + '/ratings_target.txt', 'w', encoding='utf-8')
    truthWriter = open('../data/' + DATASET + '/ratings_truth.txt', 'w', encoding='utf-8')
    simItemsWriter = open('../data/' + DATASET + '/ratings_scores_obs.txt', 'w', encoding='utf-8')

    lines = []
    for line in open(inputFile, encoding='utf-8').readlines()[1:]:
        lines.append(line)

    train_size = int(round(len(lines) * ratio, 0))
    print(train_size)
    count = 0
    order = random.sample(range(0, len(lines)), len(lines))

    for i in order:
        line = lines[i]

        array = line.strip().split('\t')

        user_index = int(array[0])
        item_index = int(array[1])
        rating = float(array[2])

        if count < train_size:
            simItemsWriter.write('%d\t%d\t%f\n' % (user_index, item_index, rating))
            if rating > 0:
                rating = 1
            else:
                rating = 0
            observationWriter.write('%d\t%d\t%f\n' % (user_index, item_index, rating))

            count = count + 1
        else:
            if rating > 0:
                rating = 1
            else:
                rating = 0
            targetWriter.write('%d\t%d\n' % (user_index, item_index))
            truthWriter.write('%d\t%d\t%f\n' % (user_index, item_index, rating))
            count = count + 1

    # inputFile.close()
    observationWriter.close()
    truthWriter.close()
    targetWriter.close()


def findkSimilarUsers(user_id, ratings, metric, k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    return similarities, indices


def get_sim_users():
    print('getting similarUsers ... ')
    simUserWriter = open('../data/' + DATASET + '/simUsers.txt', 'w', encoding='utf-8')
    k = 10
    metric = 'cosine'

    inputFile = '../data/' + DATASET + '/' + 'ratings_scores_obs.txt'

    df = pd.read_csv(inputFile, sep="\t", header=None)
    df.rename(columns={0: 'users', 1: 'items', 2: 'ratings'}, inplace=True)
    users = list(df['users'].unique())

    userItemMatrix = df.groupby(['users', 'items']).size().unstack(fill_value=0)

    for user_id in users:
        similarities, indices = findkSimilarUsers(user_id, userItemMatrix, metric, k)
        userlist = list(indices.flatten())

        for i in range(0, len(userlist)):

            if userlist[i] == user_id:
                continue
            else:
                simUserWriter.write('%d\t%d\n' % (user_id, userlist[i]))

    simUserWriter.close()


def aggregate(x):
    vc = x.value_counts()
    return vc.index[0]


def get_sim_items(userCount):
    print('getting similar Highly Rated Items and generating pairs ... ')
    simMoviesWriter = open('../data/' + DATASET + '/simRatedItems.txt', 'w', encoding='utf-8')
    inputFile = '../data/' + DATASET + '/' + 'ratings_scores_obs.txt'
    data = '../data/' + DATASET + '/' + 'ratings_scores.txt'

    user_movie_ratings = dict()

    for line in open(data, encoding='utf-8').readlines()[1:]:
        array = line.strip().split('\t')

        item_index = int(array[1])
        user_index = int(array[0])
        rating = float(array[2])

        # Storing obs user-movie ratings
        if user_index not in user_movie_ratings:
            user_movie_ratings[user_index] = set()
        user_movie_ratings[user_index].add(item_index)
        simMoviesWriter.write('%d\t%d\t1\n' % (user_index, item_index))

    df = pd.read_csv(inputFile, sep="\t", header=None)
    df.rename(columns={0: 'users', 1: 'items', 2: 'ratings'}, inplace=True)

    agg = df.loc[:, ['items', 'ratings']].groupby(['items']).agg(aggregate)

    desiredMovieIds = agg[agg['ratings'] > 0]
    itemIds = list(desiredMovieIds.index.values)
    print(user_cnt)
    print(len(itemIds))

    for user in range(0, userCount):
        for item in itemIds:
            if user in user_movie_ratings:
                items = user_movie_ratings[user]
                if item not in items:
                    simMoviesWriter.write('%d\t%d\t1\n' % (user_index, item_index))

    simMoviesWriter.close()


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    user_cnt = convert_rating()
    convert_kg()
    data_split()
    get_sim_users()
    get_sim_items(user_cnt)

    print('done')
