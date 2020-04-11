from models.Pairwise.cml import AntiSparseCML
import numpy as np
from data.core import load_mat
from eval import ndcg_at_k
import os
import tensorflow as tf
from models import restore_model
import itertools

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)

datasets = ['ml-1m', 'amazon-CDs', 'yelp']

r_list = [8, 16, 32, 64, 128, 256]
b_sizes = [512, 1024]
margins = [0.25, 0.5, 0.75, 1]
n_negs = [1, 5]
lambdas = np.logspace(-4, -1, 10)

params = {}
params['seed'] = 42
params['n_epochs'] = 151
params['lr'] = 7.5e-4
params['n_users_eval'] = 7000
params['eval_every'] = 10
data_dir =  'data'
saving_dir = 'results'

for dataset in datasets:
    mat_file = os.path.join(data_dir, dataset, 'ratings')
    train_ratings, eval_ratings, test_ratings = load_mat(mat_file)
    m, n = train_ratings.shape
    n_inters = len(train_ratings.data)
    print('Number of users/items: {}/{}'.format(m, n))
    print('Number of positive train interactions: {}'.format(n_inters))
    params['saving_dir'] = os.path.join(saving_dir, dataset)
    for r in r_list:
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('EMBEDDING DIM = {}'.format(r))
        params['r'] = r
        hp_it = itertools.product(b_sizes, n_negs, margins, lambdas)
        for b_size, n_neg, margin, lambd in hp_it:
            print('BATCH SIZE = {}'.format(b_size))
            print('N NEGATIVES = {}'.format(n_neg))
            print('MARGIN = {}'.format(margin))
            print('LAMBDA = {}'.format(lambd))
            params['batch_size'] = b_size
            params['n_negatives'] = n_neg
            params['margin'] = margin
            params['lambda'] = lambd
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                as_cml = AntiSparseCML(params)
                as_cml.fit(sess, train_ratings, eval_ratings, test_ratings, save=True)
            tf.reset_default_graph()
