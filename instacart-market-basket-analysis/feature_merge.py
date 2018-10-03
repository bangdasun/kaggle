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
    product_features = pd.read_csv('product_features.csv')
    product_w2v_features = pd.read_csv('product_w2v_features.csv')
    user_features = pd.read_csv('user_features.csv')
    user_product_features = pd.read_csv('user_x_product_features.csv')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(train.shape, test.shape)
	
	
with timer('Reduce memory usage'):
    memory_usage()
    product_features = reduce_memory_usage(product_features)
    product_w2v_features = reduce_memory_usage(product_w2v_features)
    user_features = reduce_memory_usage(user_features)
    user_product_features = reduce_memory_usage(user_product_features)
    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    memory_usage()
	
	
with timer('Merge features'):
    train = train.merge(product_features, on='product_id', how='left')
    test = test.merge(product_features, on='product_id', how='left')
    del product_features
    gc.collect()
    
    train = train.merge(product_w2v_features, on='product_id', how='left')
    test = test.merge(product_w2v_features, on='product_id', how='left')
    del product_w2v_features
    gc.collect()
    
    train = train.merge(user_features, on='user_id', how='left')
    test = test.merge(user_features, on='user_id', how='left')
    del user_features
    gc.collect()
    
    train = train.merge(user_product_features, on=['product_id', 'user_id'], how='left')
    test = test.merge(user_product_features, on=['product_id', 'user_id'], how='left')
    del user_product_features
    gc.collect()
	
	
with timer('Save train/test'):
    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    train.to_csv('train_processed_0930.csv', index=False)
    test.to_csv('test_processed_0930.csv', index=False)