
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from kaggle_learn.utils import timer

with open("df_full.pkl", "rb") as f:
    df_full = pickle.load(f)
    
with open("features_category.pkl", "wb") as f:
    features_category = pickle.load(f)
    
with open("features.pkl", "wb") as f:
    features = pickle.load(f)
    
ntrain = 307507

with timer('Prepare / Train for LightGBM'):
    print('df_full shape = {}'.format(df_full.shape))
    X_train_all = df_full.iloc[:ntrain][features].replace([np.inf, -np.inf], np.nan)
    y_train_all = df_full.iloc[:ntrain][target].replace([np.inf, -np.inf], np.nan)
    fold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    X_test = df_full.iloc[ntrain:][features].replace([np.inf, -np.inf], np.nan)
    sub = pd.read_csv('sample_submission.csv')
    oof_preds_lgb_1 = np.zeros(X_train_all.shape[0])
    sub_preds_lgb_1 = np.zeros(X_test.shape[0])
    
    for n_fold, (trn_idx, val_idx) in enumerate(fold.split(X_train_all, y_train_all)):
        X_train = X_train_all.iloc[trn_idx]
        y_train = y_train_all[trn_idx]
        
        X_val = X_train_all.iloc[val_idx]
        y_val = y_train_all[val_idx]
        
        X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=features_category)
        X_val_lgb = lgb.Dataset(X_val, y_val, feature_name=features, categorical_feature=features_category)
    
        lgb_params = {
            'objective'        : 'binary',
            'boosting_type'    : 'gbdt',
            'metric'           : 'auc',
            'data_random_seed' : 42,
            'num_leaves'       : 50,#34, #35,
            'max_depth'        : 6,#8, #-1,
            'learning_rate'    : 0.01,#.02, #.01,
            'feature_fraction' : 0.151,#.9497036, #.05,
            'bagging_fraction' : 1,#.8715623, #.9,
            'max_bin'          : 300,
            'min_child_samples': 70,
            'min_gain_to_split': 0.0222415, #0.5,
            "min_child_weight" : 39.3259775,
            'reg_lambda'       : 0.0735294, #100,
            "reg_alpha"        : 0.041545473,
            'nthread'          : 4,
        }
        
        print('='*30, ' Fold {} '.format(n_fold+1), '='*30)
        print('Total = {}, positive = {}, ratio = {:.4f}'.format(y_train.shape[0], np.sum(y_train), np.sum(y_train) / y_train.shape[0]))
        print('Total = {}, positive = {}, ratio = {:.4f}'.format(y_val.shape[0], np.sum(y_val), np.sum(y_val) / y_val.shape[0]))
        
        lgb_model = lgb.train(lgb_params
                              , train_set=X_train_lgb
                              , valid_sets=[X_train_lgb, X_val_lgb]
                              , num_boost_round=10000
                              , early_stopping_rounds=200
                              , verbose_eval=500)
        
        oof_preds_lgb_1[val_idx] = lgb_model.predict(X_val)
        sub_preds_lgb_1 += lgb_model.predict(X_test) / fold.n_splits
    
with timer('Submission file'):
    sub['TARGET'] = sub_preds_lgb_1
    sub.to_csv('sub_lgb_10f.csv', index=False)