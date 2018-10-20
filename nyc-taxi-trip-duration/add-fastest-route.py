import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/Bangda/Desktop/kaggle/ny-taxi')

train = pd.read_csv('train_featured.csv')
test  = pd.read_csv('test_featured.csv')
train.shape, test.shape

# train parts
fst_route1 = pd.read_csv('C:/Users/Bangda/Desktop/kaggle/ny-taxi/external-data/fastest_routes_train_part_1.csv',
                        usecols = ['id', 'number_of_steps', 'total_distance', 'total_travel_time'])
fst_route2 = pd.read_csv('C:/Users/Bangda/Desktop/kaggle/ny-taxi/external-data/fastest_routes_train_part_2.csv',
                        usecols = ['id', 'number_of_steps', 'total_distance', 'total_travel_time'])

# merge two parts together for train
train_route_info = pd.concat([fst_route1, fst_route2])

# test part
test_route_info  = pd.read_csv('C:/Users/Bangda/Desktop/kaggle/ny-taxi/external-data/fastest_routes_test.csv',
                              usecols = ['id', 'number_of_steps', 'total_distance', 'total_travel_time'])
train_route_info.head()

# append new info to raw data
train = train.merge(train_route_info, on = 'id', how = 'left')
test  = test.merge(test_route_info, on = 'id', how = 'left')
del fst_route1, fst_route2, train_route_info, test_route_info
train.shape, test.shape

train.columns.values

# save
train.to_csv('train_featured2.csv', index = False)
test.to_csv('test_featured2.csv', index = False)

import gc
gc.collect()