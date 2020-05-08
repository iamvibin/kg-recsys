import numpy as np
import pandas as pd
from scipy.stats import wishart
from recommend.bpmf import BPMF
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings


def file_writer(path, target_matrix, preds):
    writer = open(path, 'w')
    size = len(target_matrix)

    for i in range(0, size):
        user = target_matrix[i][0]
        item = target_matrix[i][1]
        rating = preds[i]
        writer.write('%d\t%d\t%f\n' % (user, item, rating))
# load user ratings
df = pd.read_csv('ratings_obs.txt', delimiter="\t", header=None, names=["users", "items", "ratings"])
df['ratings'] = df['ratings']*5
ratings = load_movielens_1m_ratings('ratings.dat')
df = df[['users', 'items']].astype(int)
ratings = df.values
print(ratings)

n_user = max(ratings[:, 0])+1
n_item = max(ratings[:, 1])+1
print(n_user)
print(n_item)
#ratings[:, (0, 1)] -= 1 # shift ids by 1 to let user_id & movie_id start from 0

# fit model
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=10,
                max_rating=1., min_rating=0., seed=0).fit(ratings, n_iters=2)
RMSE(bpmf.predict(ratings[:, :2]), ratings[:,2]) # training RMSE

df = pd.read_csv('ratings_target.txt', delimiter="\t", header=None, names=["users", "items"])
df = df[list(df.columns)].astype(int)
target_matrix = df.values
# predict ratings for user 0 and item 0 to 9:
preds = bpmf.predict(target_matrix)
#preds = preds/5
print(preds)
file_writer('baseline_output.txt', target_matrix, preds)
#print((bpmf.predict(np.array([[0, i] for i in range(10)]))))