import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer, reduce_memory_usage, memory_usage


with timer('Load data'):
    aisles = pd.read_csv('aisles.csv')
    departments = pd.read_csv('departments.csv')
    order_products_prior = pd.read_csv('order_products__prior.csv')
    order_products_train = pd.read_csv('order_products__train.csv')
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
	
	
with timer('Get prior/train/test data'):
    sub = pd.read_csv('sample_submission.csv')
    prior = orders.loc[orders['eval_set'] == 'prior']
    train = orders.loc[orders['eval_set'] == 'train']
    test = orders.loc[orders['eval_set'] == 'test']
    print(prior.shape, train.shape, test.shape, sub.shape)
	

with timer('Process prior/train/test'):
    prior = order_products_prior.merge(prior, on=['order_id'], how='left')
    
    # get the train users all product history, later a binary classification will be applied on it than we can convert to the submission format 
    train_user_product_history = prior.loc[prior['user_id'].isin(train['user_id'].unique()), ['user_id', 'product_id']].drop_duplicates(keep='first')
    train = train.merge(train_user_product_history, on=['user_id'], how='right')
    del train_user_product_history
    gc.collect()
    
    # get the test users all product history, later a binary classification will be applied on it than we can convert to the submission format 
    test_user_product_history = prior.loc[prior['user_id'].isin(test['user_id'].unique()), ['user_id', 'product_id']].drop_duplicates(keep='first')
    test = test.merge(test_user_product_history, on=['user_id'], how='right')
    del test_user_product_history
    gc.collect()
    
    train = train.merge(order_products_train, on=['order_id', 'product_id'], how='left')
    
    print(prior.shape, train.shape, test.shape)
	

with timer('Save transformed data'):
    del prior['eval_set']
    del train['eval_set']
    del test['eval_set']
    gc.collect()
    prior = reduce_memory_usage(prior)
    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    
    prior.to_csv('prior.csv', index=False)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
	
