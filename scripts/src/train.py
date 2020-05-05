import tensorflow as tf
import numpy as np
from src.model import MKR
import os


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]

    BASELINE_OUTPUT_FILE = 'baseline_output.txt'
    RESULT_OUTPUT_FILE = 'result.txt'
    LABELS_FILE = 'labels.txt'
    SCORES_FILE = 'scores.txt'
    dir_name = args.out + '_' + str(args.i)

    RESULT_OUTPUT_PATH = os.path.join('..', 'data', args.dataset, dir_name, RESULT_OUTPUT_FILE)
    tf.reset_default_graph()
    model = MKR(args, n_user, n_entity, n_entity, n_relation)

    # top-K evaluation settings
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            # RS training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train_rs(sess, get_feed_dict_for_rs(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(loss)

            # KGE training
            if step % args.kge_interval == 0:
                np.random.shuffle(kg)
                start = 0
                while start < kg.shape[0]:
                    _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(rmse)

            # CTR evaluation
            train_auc, train_acc = model.eval(sess, get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(sess, get_feed_dict_for_rs(model, eval_data, 0, eval_data.shape[0]))
            test_auc, test_acc = model.eval(sess, get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))

            # print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
            #     % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            # top-K evaluation
            if show_topk:
                precision, recall, f1 = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('f1: ', end='')
                for i in f1:
                    print('%.4f\t' % i, end='')
                print('\n')

        print("Writing output of baseline for split %s ." % args.i)
        labels, scores = sess.run([model.labels, model.scores_normalized],
                                  get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))

        LABELS_OUTPUT_PATH = os.path.join('..', 'data', args.dataset, dir_name, LABELS_FILE)
        SCORES_OUTPUT_PATH = os.path.join('..', 'data', args.dataset, dir_name, SCORES_FILE)
        with open(LABELS_OUTPUT_PATH, 'w') as labelsWriter:
            for i in labels:
                labelsWriter.write(str(i) + '\n')

        with open(SCORES_OUTPUT_PATH, 'w') as scoreWriter:
            for i in scores:
                scoreWriter.write(str(i) + '\n')
        print('Baseline scores and labels written .')
        '''
        with open(RESULT_OUTPUT_PATH, 'w') as result_writer:
            result_writer.write('train_auc\t%.4f\n' % (train_auc))
            result_writer.write('eval_auc\t%.4f\n' % (eval_auc))
            result_writer.write('test_auc\t%.4f\n' % (test_auc))
            result_writer.write('precision\t')
            for ele in precision:
                result_writer.write(str(ele) + '\t')
            result_writer.write('\n')

            result_writer.write('recall\t')
            for ele in recall:
                result_writer.write(str(ele) + '\t')
            result_writer.write('\n')

            result_writer.write('f1\t')
            for ele in f1:
                result_writer.write(str(ele) + '\t')
            result_writer.write('\n')

        result_writer.close()'''

        neighbours = args.list

        for neighbour in neighbours:

            n_dir_name = str(neighbour).zfill(3) + '_' + str(args.i)
            print('Writing Baseline output for split %s' % n_dir_name)
            TARGET_FILE = 'ratings_target.txt'
            TARGET_FILE_PATH = os.path.join('..', 'data', args.dataset, n_dir_name, TARGET_FILE)

            user_set = set()
            user_item_dict = {}
            with open(TARGET_FILE_PATH, 'r', encoding="utf-8") as file:
                for line in file:
                    array = line.strip().split('\t')
                    user = int(array[0])
                    item = int(array[1])
                    user_set.add(user)
                    if user not in user_item_dict:
                        user_item_dict[user] = set()
                    user_item_dict[user].add(item)
            print('Baseline output data loaded for split %s.' % n_dir_name)

            user_list = list(user_set)
            OUTPUT_PATH = os.path.join('..', 'data', args.dataset, n_dir_name, BASELINE_OUTPUT_FILE)
            with open(OUTPUT_PATH, 'w') as writer:
                for user in user_list:

                    test_item_list = list(user_item_dict[user])
                    items, scores = model.get_scores(sess, {model.user_indices: [user] * len(test_item_list),
                                                            model.item_indices: test_item_list,
                                                            model.head_indices: test_item_list})
                    for item, score in zip(items, scores):
                        writer.write('%d\t%d\t%f\n' % (user, item, score))
                print('baseline output for %s finished.' % n_dir_name)


def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.head_indices: data[start:end, 1]}
    return feed_dict


def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {model.item_indices: kg[start:end, 0],
                 model.head_indices: kg[start:end, 0],
                 model.relation_indices: kg[start:end, 1],
                 model.tail_indices: kg[start:end, 2]}
    return feed_dict


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        items, scores = model.get_scores(sess, {model.user_indices: [user] * len(test_item_list),
                                                model.item_indices: test_item_list,
                                                model.head_indices: test_item_list})
        for item, score in zip(items, scores):
            item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

    return precision, recall, f1


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
