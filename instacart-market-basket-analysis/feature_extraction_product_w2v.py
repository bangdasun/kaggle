# This is from public kernel by @omarito
# https://www.kaggle.com/omarito/word2Vec-for-products-analysis-0-01-lb

import os
import pickle
import numpy as np
import pandas as pd
import gensim

from kaggle_learn.utils import *
from sklearn.decomposition import PCA


with timer('Load data'):
    prior = pd.read_csv('prior.csv')
    train = pd.read_csv('train.csv')
	

with timer('Select product_id and order_id'):
    prior = prior[['order_id', 'product_id']]
    train = train[['order_id', 'product_id']]
	

with timer('Convert product_id into sentences indexed by order_id'):
    prior['product_id'] = prior['product_id'].astype(str)
    train['product_id'] = train['product_id'].astype(str)
    
    prior = prior.groupby('order_id').apply(lambda gp: gp['product_id'].tolist())
    train = train.groupby('order_id').apply(lambda gp: gp['product_id'].tolist())
	

with timer('Train Word2Vec'):
    corpus = pd.concat([prior, train], axis=0).values
    model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)
	

with timer('Get the product embeddings'):
    vocab_size = len(model.wv.vocab)
    embedding_matrix = np.zeros((vocab_size, 100))
    for i in range(1, vocab_size + 1):
        w = str(i)
        if w in model.wv.vocab:
            embedding_matrix[i - 1] = model.wv[w]
			
	
with timer('Dimension reduction use PCA'):
    features_num = 5
    pca = PCA(features_num)
    embedding_matrix = pca.fit_transform(embedding_matrix)
    w2v_features = pd.DataFrame(embedding_matrix)
    w2v_features.columns = ['product_w2v_{}'.format(i) for i in range(features_num)]
    w2v_features['product_id'] = range(1, vocab_size + 1)
    print(w2v_features.head())
	
	
with timer('Save features'):
    w2v_features.to_csv('product_w2v_features.csv', index=False)
