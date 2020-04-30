import numpy as np
import os
import random

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})
entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()

def read_item_index_to_entity_id_file(DATASET):
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating(DATASET):
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET.split('/')[0]]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    user_movie_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[0:]:
        array = line.strip().split(SEP[DATASET.split('/')[0]])

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

        if rating >= THRESHOLD[DATASET.split('/')[0]]:
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


def convert_kg(DATASET):
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/relations_obs.txt', 'w', encoding='utf-8')
    kg_writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
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
        kg_writer.write('%d\t%d\t%d\n' % (head, relation, tail))
        writer.write('%d\t%d\t%d\n' % (tail, relation, head))

    writer.close()
    kg_writer.close()
    file.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def data_split(DATASET):
    print('spliting data into observed and target ...')
    ratio = 0.8

    inputFile = '../data/' + DATASET + '/' + 'ratings_scores.txt'

    observationWriter = open('../data/' + DATASET + '/ratings_obs.txt', 'w', encoding='utf-8')
    targetWriter = open('../data/' + DATASET + '/ratings_target.txt', 'w', encoding='utf-8')
    truthWriter = open('../data/' + DATASET + '/ratings_truth.txt', 'w', encoding='utf-8')
    simItemsWriter = open('../data/' + DATASET + '/ratings_scores_obs.txt', 'w', encoding='utf-8')

    lines = []
    for line in open(inputFile, encoding='utf-8').readlines()[0:]:
        lines.append(line)
    print(len(lines))

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
    simItemsWriter.close()


def get_relations(DATASET):
    print('Converting kg.txt into its respective relation files.')
    FOLDER = '../data'
    DATASET = 'movie'
    RELATIONS = 'relations'
    INPUT_FILE = 'kg.txt'
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    input_path = os.path.join(FOLDER, DATASET, INPUT_FILE)
    output_folder_path = os.path.join(FOLDER, DATASET, RELATIONS)

    relations_dict = {}
    relations_map = {}
    file_writer = {}
    print('creating a file writer set')
    with open(input_path, 'r', encoding="utf-8") as file:
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

            filename = relation_old.strip().split('.')[2]

            if filename not in relations_map:
                relations_map[relation] = filename
                output_file = filename + '.txt'
                output_file_path = os.path.join(output_folder_path, output_file)
                writer = open(output_file_path, 'w', encoding='utf-8')
                file_writer[relation] = writer

            if filename not in relations_dict:
                relations_dict[filename] = set()

            text = str(head) + '\t' + str(tail)
            relations_dict[filename].add(text)

    for relation, writer in file_writer.items():
        filename = relations_map[relation]
        wr = file_writer[relation]
        text_set = relations_dict[filename]
        print('No of lines being written %s', filename, len(text_set))
        for text in text_set:
            h2t = text + '\n'
            a = text.strip().split('\t')
            t2h = a[1] + '\t' + a[0] + '\n'
            wr.write(h2t)
            wr.write(t2h)
        wr.close()


def preprocess(args):
    np.random.seed(args.s)
    DATASET = args.d

    read_item_index_to_entity_id_file(DATASET)
    user_cnt = convert_rating(DATASET)
    convert_kg(DATASET)
    data_split(DATASET)
    #get_relations()

    print('done')
