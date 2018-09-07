
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, KFold
from kaggle_learn.utils import timer


with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    df_text_processed = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('categorical_features.pkl', 'rb') as f:
    categorical_features = pickle.load(f)

with open('df_reduced.pkl', 'rb') as f:
    df_reduced = pickle.load(f)

with timer('Prepare for lightgbm'):
    
    X_train_all_dense = csr_matrix(df_reduced[features].iloc[:ntrain].values)
    X_train_all = hstack([X_train_all_dense, df_text_processed[:ntrain]])
    del X_train_all_dense
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=.1, random_state=42)
    
    X_test = hstack([df_reduced[features].iloc[ntrain:].values, df_text_processed[ntrain:]])
    
    all_features = features + vocab
    X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=all_features, categorical_feature=categorical_features)
    X_val_lgb = lgb.Dataset(X_val, y_val, feature_name=all_features, categorical_feature=categorical_features)
    del X_train_all, df_reduced, df_text_processed
    gc.collect()

with timer('Train lightgbm'):
    lgb_params = {
        'objective'       : 'regression',
        'boosting_type'   : 'gbdt',
        'metric'          : 'rmse',
        'learning_rate'   : 0.01,
        'max_depth'       : 16,
        'num_leaves'      : 2**9-1,
        'feature_fraction': 0.15,
        'bagging_fraction': 0.8,
        'data_random_seed': 42
    }
    
    lgb_model = lgb.train(lgb_params
                          , train_set=X_train_lgb
                          , valid_sets=[X_train_lgb, X_val_lgb]
                          , num_boost_round=50000
                          , early_stopping_rounds=300
                          , verbose_eval=200)

with timer('predicting'):
    sub = pd.read_csv('sample_submission.csv')
    sub['deal_probability'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    try:
        sub['deal_probability'] = np.clip(sub['deal_probability'], 0., 1.)
    except:
        print('Clip cannot be applied')

with timer('Submission'):
    sub.to_csv('sub_lgbm.csv', index=False)
