import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle_learn.utils import timer, reduce_memory_usage, memory_usage
from kaggle_learn.feature_engineering.statistics import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from instacart_utils import *

%matplotlib inline


with timer('Load data'):
    aisles = pd.read_csv('aisles.csv')
    departments = pd.read_csv('departments.csv')
    products = pd.read_csv('products.csv')
    prior = pd.read_csv('prior.csv')
    print(prior.shape)
	

with timer('Reduce memory usage'):
    memory_usage()
    prior = reduce_memory_usage(prior)
    memory_usage()
	

with timer('Product features'):
    
    # number of being ordered, number of users order the product, and the ratio
    prior = add_group_count(prior, cols=['product_id'], cname='product_bought_count', value='user_id')
    prior = add_group_nunique(prior, cols=['product_id'], cname='product_bought_user_count', value='user_id')
    prior['product_relative_popularity'] = prior['product_bought_user_count'] / (prior['product_bought_count'] + 1)
    
    # number of being reordered, and the ratio with number of being ordered
    prior = add_group_sum(prior, cols=['product_id'], cname='product_reordered_count', value='reordered')
    prior['product_reordered_ratio'] = prior['product_reordered_count'] / prior['product_bought_count']
    
    # number of being 1st time ordered, 2nd time ordered, and the ratio
    prior = add_group_cumcount(prior, cols=['user_id', 'product_id'], cname='user_buy_product_cumcount')
    prior['is_product_1st_time_bought'] = (prior['user_buy_product_cumcount'] == 1)
    prior = add_group_sum(prior, cols=['product_id'], cname='product_1st_time_bought_count', value='is_product_1st_time_bought')
    prior['is_product_2nd_time_bought'] = (prior['user_buy_product_cumcount'] == 2)
    prior = add_group_sum(prior, cols=['product_id'], cname='product_2nd_time_bought_count', value='is_product_2nd_time_bought')
    prior['product_reordered_prob'] = prior['product_2nd_time_bought_count'] / prior['product_1st_time_bought_count']
    prior['product_reordered_times'] = 1 + prior['product_bought_count'] / prior['product_1st_time_bought_count']
    
    # mean and std of product ordered dow and hod
    prior = add_group_mean(prior, cols=['product_id'], cname='product_dow_mean', value='order_dow')
    prior = add_group_std(prior, cols=['product_id'], cname='product_dow_std', value='order_dow')
    prior = add_group_mean(prior, cols=['product_id'], cname='product_hod_mean', value='order_hour_of_day')
    prior = add_group_std(prior, cols=['product_id'], cname='product_hod_std', value='order_hour_of_day')
    
    prior = add_group_mean(prior, cols=['product_id'], cname='product_add_to_cart_order_mean', value='add_to_cart_order')
    
    # features refer from: https://github.com/KazukiOnodera/Instacart/tree/master/py_feature
    # 1. how many users buy it as 'one-shot' item
    product_user_bought_once_count = (prior.groupby(['product_id', 'user_id'])['order_id'].count() == 1).reset_index().groupby(['product_id']).sum().reset_index()
    product_user_bought_once_count = product_user_bought_once_count.drop(['user_id'], axis=1)
    product_user_bought_once_count.columns = ['product_id', 'product_user_bought_once_count']
    prior = prior.merge(product_user_bought_once_count, on=['product_id'], how='left')
    prior['product_user_bought_once_count_ratio'] = prior['product_user_bought_once_count'] / prior['product_bought_user_count']
    del product_user_bought_once_count
    gc.collect()
	
	
with timer('Reduce prior memory use'):
    prior = reduce_memory_usage(prior)
    print(prior.shape)
	

with timer('Generate product features table'):
    product_features = prior.drop(['order_id', 'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day',
                                   'user_id', 'order_number', 'days_since_prior_order', 'user_buy_product_cumcount',
                                   'is_product_1st_time_bought', 'is_product_2nd_time_bought'], axis=1)
    product_features = product_features.drop_duplicates(keep='first')
    product_features = reduce_memory_usage(product_features)
    print(product_features.shape)
	

with timer('Save product feature table'):
    product_features.to_csv('product_features.csv', index=False)
