import os

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})
entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()

def read_item_index_to_entity_id_file(args):

    input_file = 'item_index2entity_id.txt'
    path = os.path.join('..', 'data', args.d, input_file)
    print('reading item index to entity id file: ' + input_file + ' ...')
    i = 0
    for line in open(path, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def get_rating(args):
    input_file = RATING_FILE_NAME[args.d]
    path = os.path.join('..', 'data', args.d, input_file)
    output_file = 'ratings.txt'
    ratings_path = os.path.join('..', 'data', args.d, output_file)
    max_rating = 10 # 5 for movie

    print('Converting rating file ...')

    user_movie_ratings = dict()

    for line in open(path, encoding='utf-8').readlines()[0:]:
        array = line.strip().split(SEP[args.d])

        # remove prefix and suffix quotation marks for BX dataset
        if args.d == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])/max_rating

        # Storing user-movie ratings
        if user_index_old not in user_movie_ratings:
            user_movie_ratings[user_index_old] = {}
        user_movie_ratings[user_index_old][item_index] = rating

    print('converting rating file ...')
    user_cnt = 0
    user_index_old2new = dict()
    with open(ratings_path, 'w') as writer:
        for user_index_old, item_dict in user_movie_ratings.items():
            if user_index_old not in user_index_old2new:
                user_index_old2new[user_index_old] = user_cnt
                user_cnt += 1
            user_index = user_index_old2new[user_index_old]

            for item, rating in item_dict.items():
                writer.write('%d\t%d\t%f\n' % (user_index, item, rating))

    item_set = set(item_index_old2new.values())
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    return user_cnt


def convert_kg(args):
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    relations_path = os.path.join('..', 'data', args.d, 'relations_obs.txt')
    kg_input_path = os.path.join('..', 'data', args.d, 'kg.txt')
    writer = open(relations_path, 'w', encoding='utf-8')

    file = open(kg_input_path, encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            if tail_old not in entity_id2index:
                continue

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
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
    file.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)

def reformat(args):
    read_item_index_to_entity_id_file(args)
    user_cnt = get_rating(args)
    convert_kg(args)
    return entity_id2index ,relation_id2index, item_index_old2new, user_cnt