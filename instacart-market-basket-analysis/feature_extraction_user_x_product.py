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
	
	
with timer('Simple features'):
    prior = add_group_mean(prior, cols=['order_id'], cname='order_product_reordered_mean', value='reordered')
    prior = add_group_max(prior, cols=['user_id'], cname='user_order_number_max', value='order_number')
    prior['user_product_id'] = prior['product_id'] + prior['user_id'] * 1e5
    prior['order_number_desc'] = prior['user_order_number_max'] - prior['order_number'] - 1
    prior['user_product_decay'] = 0.8 ** prior['order_number_desc']
    prior['user_product_add_cart_order_decay'] = 0.99 ** prior['add_to_cart_order'] * prior['user_product_decay']
	
	
with timer('User x Product features'):
    # already in user features
    prior = add_group_max(prior, cols=['user_id'], cname='user_order_count', value='order_number')
    
    # number of this product being ordered by this user
    prior = add_group_count(prior, cols=['user_product_id'], cname='user_product_bought_count', value='order_number')
    
    # first order and last order
    prior = add_group_min(prior, cols=['user_product_id'], cname='user_product_bought_1st_order', value='order_number')
    prior = add_group_max(prior, cols=['user_product_id'], cname='user_product_bought_last_order', value='order_number')
    prior['user_product_ordered_rate'] = prior['user_product_bought_count'] / prior['user_order_count']
    prior['user_product_order_count_since_last_order'] = prior['user_order_count'] - prior['user_product_bought_last_order']
    prior['user_product_order_rate_since_1st_order'] = prior['user_product_bought_count'] / (prior['user_order_count'] - prior['user_product_bought_1st_order'] + 1)
    
    # add_to_cart mean
    prior = add_group_mean(prior, cols=['user_product_id'], cname='user_product_cart_position_mean', value='add_to_cart_order')
    
    # statistics of user_product_decay
    prior = add_group_sum(prior, cols=['user_product_id'], cname='user_product_decay_sum', value='user_product_decay')
    
    # statistics of order datetime
    prior = add_group_mean(prior, cols=['user_product_id', 'order_dow'], cname='user_product_dow_reordered_mean', value='reordered')
    prior = add_group_mean(prior, cols=['user_product_id', 'order_hour_of_day'], cname='user_product_hod_reordered_mean', value='reordered')
    
    # user product reordered mean / sum given by order_id
    prior = add_group_mean(prior, cols=['user_product_id'], cname='user_product_order_id_reordered_mean', value='order_product_reordered_mean')
    prior = add_group_sum(prior, cols=['user_product_id'], cname='user_product_order_id_reordered_sum', value='order_product_reordered_mean')
	
	
with timer('Reduce prior memory use'):
    prior = reduce_memory_usage(prior)
    print(prior.shape)
	

with timer('Generate user x product features table'):
    userxproduct_features = prior.drop(['order_id', 'add_to_cart_order', 'reordered', 'order_dow',
                                        'order_hour_of_day', 'order_number', 'days_since_prior_order',
                                        'order_number_desc', 'user_product_decay', 'user_product_add_cart_order_decay',
                                        'user_order_count', 'order_product_reordered_mean', 'user_order_number_max',
                                        'user_product_id', 'user_product_dow_reordered_mean', 'user_product_hod_reordered_mean'], axis=1)
    userxproduct_features = userxproduct_features.drop_duplicates(keep='first')
    userxproduct_features = reduce_memory_usage(userxproduct_features)
    print(userxproduct_features.shape)
	
	
with timer('Save user x product features table'):
    userxproduct_features.to_csv('user_x_product_features.csv', index=False)
