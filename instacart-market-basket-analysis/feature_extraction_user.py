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
	

with timer('User features'):
    # number of user orders
    prior = add_group_max(prior, cols=['user_id'], cname='user_order_count', value='order_number')
    
    # number of user order product, unique product, reorder product
    prior = add_group_count(prior, cols=['user_id'], cname='user_products_count', value='product_id')
    prior = add_group_nunique(prior, cols=['user_id'], cname='user_products_nunique', value='product_id')
    prior = add_group_sum(prior, cols=['user_id'], cname='user_reordered_sum', value='reordered')
    gp = prior.groupby(['user_id', 'product_id'])['reordered'].sum().reset_index()
    gp['is_reordered_product'] = (gp['reordered'] > 1).astype(int)
    gp = gp.groupby('user_id')['is_reordered_product'].sum()
    gp_df = pd.DataFrame(gp.values, columns=['user_product_reordered_nunique'])
    gp_df['user_id'] = gp.index
    prior = prior.merge(gp_df, on='user_id', how='left')
    del gp_df
    gc.collect()
    prior['user_product_reordered_ratio'] = prior['user_product_reordered_nunique'] / prior['user_products_nunique']
    
    # user reordered
    prior['is_not_1st_order'] = prior['order_number'] > 1
    prior = add_group_sum(prior, cols=['user_id'], cname='user_is_not_1st_order_count', value='is_not_1st_order')
    prior['user_reordered_ratio'] = prior['user_reordered_sum'] / prior['user_is_not_1st_order_count']
    
    # statistics of basket size
    prior['user_basket_size_mean'] = prior['user_products_count'] / prior['user_order_count']
    prior['user_basket_size_unique_mean'] = prior['user_products_nunique'] / prior['user_order_count']
    prior['user_basket_size_unique_ratio'] = prior['user_basket_size_unique_mean'] / prior['user_basket_size_mean']
    
    # statistics of user days since prior order (interval of different orders)
    prior = add_group_sum(prior, cols=['user_id'], cname='user_days_since_prior_order_sum', value='days_since_prior_order')
    prior = add_group_mean(prior, cols=['user_id'], cname='user_days_since_prior_order_mean', value='days_since_prior_order')
    prior = add_group_min(prior, cols=['user_id'], cname='user_days_since_prior_order_min', value='days_since_prior_order')
    prior = add_group_max(prior, cols=['user_id'], cname='user_days_since_prior_order_max', value='days_since_prior_order')
    prior = add_group_std(prior, cols=['user_id'], cname='user_days_since_prior_order_std', value='days_since_prior_order')
	
	
with timer('Reduce prior memory use'):
    prior = reduce_memory_usage(prior)
    print(prior.shape)
	

with timer('Generate user features table'):
    user_features = prior.drop(['order_id', 'product_id', 'add_to_cart_order', 'reordered',
                                'order_dow', 'order_hour_of_day', 'order_number',
                                'days_since_prior_order', 'is_not_1st_order'], axis=1)
    user_features = user_features.drop_duplicates(keep='first')
    user_features = reduce_memory_usage(user_features)
    print(user_features.shape)
	

with timer('Save user features table'):
    user_features.to_csv('user_features.csv', index=False)
