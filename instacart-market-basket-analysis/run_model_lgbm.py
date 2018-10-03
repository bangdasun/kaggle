import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from tqdm import tqdm
from kaggle_learn.utils import timer, reduce_memory_usage, memory_usage
from kaggle_learn.feature_engineering.statistics import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from instacart_utils import *

%matplotlib inline


with timer('Load data'):
    train = pd.read_csv('train_processed_0930.csv')
    test = pd.read_csv('test_processed_0930.csv')
    print(train.shape, test.shape)
	

with timer('Prepare features'):
    features = test.columns.tolist()
    features.remove('order_id')
    features.remove('user_id')
    features.remove('product_id')
    features.remove('order_number')
    features.remove('user_is_not_1st_order_count')
    categorical_features = ['order_dow', 'order_hour_of_day']
    print('Number of features = {}'.format(len(features)))
	
	
with timer('Train lightgbm'):
    lgb_params = {
        'objective'       : 'binary',
        'boosting_type'   : 'gbdt',
        'metric'          : 'binary_logloss',
        'learning_date'   : 0.02,
        'max_depth'       : 8,
        'num_leaves'      : 32,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.8,
        'data_random_seed': 42,
    }
    
    X_train_all = train[features].values
    y_train_all = train['reordered'].fillna(value=0.0).values
    X_test = test[features].values
    sub_preds_lgb = np.zeros(X_test.shape[0])
    
    cv = True
    if cv:
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        for n_fold, (trn_idx, val_idx) in enumerate(fold.split(X_train_all, y_train_all)):
            X_train = X_train_all[trn_idx]
            X_val = X_train_all[val_idx]
            y_train = y_train_all[trn_idx]
            y_val = y_train_all[val_idx]

            X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=categorical_features)
            X_val_lgb = lgb.Dataset(X_val, y_val, feature_name=features, categorical_feature=categorical_features)

            print('='*30, f' Fold {n_fold + 1} ', '='*30)
            lgb_model = lgb.train(lgb_params, train_set=X_train_lgb, valid_sets=[X_train_lgb, X_val_lgb],
                                  num_boost_round=10000, early_stopping_rounds=200, verbose_eval=200)
            sub_preds_lgb += lgb_model.predict(X_test) / fold.n_splits
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, random_state=42, test_size=0.2)
        X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=categorical_features)
        X_val_lgb = lgb.Dataset(X_val, y_val, feature_name=features, categorical_feature=categorical_features)

        lgb_model = lgb.train(lgb_params, train_set=X_train_lgb, valid_sets=[X_train_lgb, X_val_lgb],
                              num_boost_round=10000, early_stopping_rounds=100, verbose_eval=200)
        sub_preds_lgb += lgb_model.predict(X_test)
		
		
with timer('Plot feature importance'):
    f, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(lgb_model, ax=ax, importance_type='gain')

	
with timer('Clean variables'):
    try:
        del train, X_train_all, y_train_all, X_train, y_train, X_val, y_val
        del X_train_lgb, X_val_lgb
        gc.collect()
    except:
        pass
		
		
with timer('Maximum F1-score (Processed submission)'):
    test['prediction'] = sub_preds_lgb
    test_sub = test.loc[test['prediction'] > 0.01, ['order_id', 'product_id', 'prediction']]
    out = [create_products_faron(gp) for name, gp in tqdm(test_sub.groupby(['order_id']))]
	
	
with timer('Processed submission'):
    sub = pd.DataFrame(data=out, columns=['order_id', 'products'])
    print(sub.shape)
    sub.to_csv('sub_lgb_v12_5f_processed.csv', index=False)
	
	
with timer('Normal submission'):
    test['reordered'] = (sub_preds_lgb > 0.18).astype(int)
    test['product_id'] = test['product_id'].astype(str)
    sub = pd.read_csv('sample_submission.csv')
    sub = sub.merge(test.loc[test['reordered'] == 1, ['order_id', 'product_id', 'reordered']], 
                    on=['order_id'], how='left')
    sub.fillna(value='None', inplace=True)
    sub['products'] = sub.groupby(['order_id'])['product_id'].transform(lambda x: ' '.join(x))
    sub = sub.drop(['product_id', 'reordered'], axis=1).drop_duplicates(keep='first')
    print(sub.shape)
    sub.to_csv('sub_lgb_v7.csv', index=False)