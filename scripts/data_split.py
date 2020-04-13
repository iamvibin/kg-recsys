import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d
    ratio = 0.8

    inputFile = '../data/' + DATASET + '/' + 'ratings_final.txt'

    observationWriter = open('../data/' + DATASET + '/ratings_obs.txt', 'w', encoding='utf-8')
    targetWriter = open('../data/' + DATASET + '/ratings_target.txt', 'w', encoding='utf-8')
    truthWriter = open('../data/' + DATASET + '/ratings_truth.txt', 'w', encoding='utf-8')

    lines = []
    for line in open(inputFile, encoding='utf-8').readlines()[1:]:
        lines.append(line)

    train_size = int(round(len(lines) * ratio, 0))
    count = 0
    order = random.sample(range(0, train_size), train_size)

    for i in order:
        line = lines[i]

        array = line.strip().split('\t')

        user_index = int(array[0])
        item_index = int(array[1])
        rating = float(array[2])

        if count < train_size:
            observationWriter.write('%d\t%d\t%f\n' % (user_index, item_index, rating))
            count = count + 1
        else:
            targetWriter.write('%d\t%d\n' % (user_index, item_index))
            truthWriter.write('%d\t%d\t%f\n' % (user_index, item_index, rating))

    inputFile.close()
    observationWriter.close()
    truthWriter.close()
    targetWriter.close()
