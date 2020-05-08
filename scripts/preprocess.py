import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})
entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()


def readfile(path):
    df = pd.read_csv(path, delimiter="\t", header=None, names=["users", "items", "ratings"])
    return df


def file_writer(file_path, df_x, df_y, size, file_type):
    writer = open(file_path, 'w')
    x_matrix = df_x.values
    y_matrix = df_y.values

    for i in range(0, size):
        user = x_matrix[i][0]
        item = x_matrix[i][1]
        rating = y_matrix[i]
        if file_type:
            writer.write('%d\t%d\t%f\n' % (user, item, rating))
        else:
            writer.write('%d\t%d\n' % (user, item))

    writer.close()


def data_split(args):
    print('spliting data ')

    dir_name = os.path.join('..', 'data', args.d)
    split_dir_name = args.out+'_'+str(args.i)

    input_file = 'ratings.txt'
    train_file = 'ratings_obs.txt'
    truth_file = 'ratings_truth.txt'
    test_file = 'ratings_target.txt'

    input_file_path = os.path.join(dir_name, input_file)

    train_file_path = os.path.join(dir_name, split_dir_name, 'eval', train_file)
    target_file_path = os.path.join(dir_name, split_dir_name, 'eval', test_file)
    truth_file_path = os.path.join(dir_name, split_dir_name, 'eval', truth_file)

    val_train_file_path = os.path.join(dir_name, split_dir_name, 'train', train_file)
    val_target_file_path = os.path.join(dir_name, split_dir_name, 'train', test_file)
    val_truth_file_path = os.path.join(dir_name, split_dir_name, 'train', truth_file)

    df = readfile(input_file_path)
    y = df['ratings']
    X = df.drop('ratings', axis=1)

    x, x_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=args.s)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=args.s)

    train_size = len(x_train)
    test_size = len(x_val)

    file_writer(val_train_file_path, x_train, y_train, train_size, True)
    file_writer(val_target_file_path, x_val, y_val, test_size, False)
    file_writer(val_truth_file_path, x_val, y_val, test_size, True)

    train_size = len(x)
    test_size = len(x_test)

    file_writer(train_file_path, x, y, train_size, True)
    file_writer(target_file_path, x_test, y_test, test_size, False)
    file_writer(truth_file_path, x_test, y_test, test_size, True)


def generate_similar_pair(args):
    dir_name = os.path.join('..', 'data', args.d)
    input_file = 'ratings.txt'

    ratings_path = os.path.join(dir_name, input_file)
    df = readfile(ratings_path)
    neighbour = args.n

    SIM_USER_OUTPUT_FILE = 'SIMILAR_USERS_'+str(neighbour)+'.txt'
    SIM_ITEM_OUTPUT_FILE = 'SIMILAR_ITEMS_'+str(neighbour)+'.txt'

    uu_outputpath = os.path.join('..', 'data', args.d, SIM_USER_OUTPUT_FILE)
    ii_outputpath = os.path.join('..', 'data', args.d, SIM_ITEM_OUTPUT_FILE)

    R_df = df.pivot(index='users', columns='items', values='ratings').fillna(0)

    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)

    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=neighbour)

    user_sim_matrix = np.dot(U, np.transpose(U))
    item_sim_matrix = np.dot(np.transpose(Vt), Vt)

    idx = -neighbour-1
    similar_users = np.argpartition(user_sim_matrix, np.argmax(user_sim_matrix, axis=0))[:, idx:]
    similar_items = np.argpartition(item_sim_matrix, np.argmax(item_sim_matrix, axis=0))[:, idx:]

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
    data_split(args)
    #get_relations()

    print('done')
