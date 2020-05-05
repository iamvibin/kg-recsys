import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, kg


def load_rating(args):
    print('reading rating file ...')
    dir_name = args.out+'_'+str(args.i)

    # reading rating file
    rating_file = os.path.join('..', 'data', args.dataset, dir_name, 'ratings_final.txt')
    train_eval_file = os.path.join('..', 'data', args.dataset, dir_name, 'ratings_obs.txt')
    test_file = os.path.join('..', 'data', args.dataset, dir_name, 'ratings_truth.txt')
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file, dtype=np.int32)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))

    print('splitting dataset into train, eval & test...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.25

    train_eval_data = np.loadtxt(train_eval_file, dtype=np.int32)
    test_data = np.loadtxt(test_file, dtype=np.int32)
    train_eval_n = train_eval_data.shape[0]
    eval_indices = np.random.choice(list(range(train_eval_n)), size=int(train_eval_n * eval_ratio), replace=False)
    eval_data = train_eval_data[eval_indices]
    train_indices = list(set(range(train_eval_n)) - set(eval_indices))
    train_data = train_eval_data[train_indices]

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    dir_name = args.out + '_' + str(args.i)
    kg_file = os.path.join('..', 'data', args.dataset, dir_name, 'kg_final.txt')
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file, dtype=np.int32)
        #np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    return n_entity, n_relation, kg