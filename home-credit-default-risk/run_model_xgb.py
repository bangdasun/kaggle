
import gc
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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

with timer('Prepare / Train for XGBoost'):
    features_numeric = list(set(features) - set(features_category))
    oh_encoder = OneHotEncoder()
    sparse_df = oh_encoder.fit_transform(df_full[features_category].values)
    sparse_df = csr_matrix(hstack((sparse_df, df_full[features_numeric].replace([np.inf, -np.inf], np.nan))))
    sparse_train_all = sparse_df[:ntrain]
    sparse_test = sparse_df[ntrain:]
    del sparse_df; gc.collect()

with timer('Prepare / Train for XGBoost'):
    print('df_full shape = {}'.format(df_full.shape))
    
    
    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    X_test = df_full.iloc[ntrain:][features].replace([np.inf, -np.inf], np.nan)
    sub = pd.read_csv('sample_submission.csv')
    
    oof_preds_xgb_1 = np.zeros(ntrain)
    sub_preds_xgb_1 = np.zeros(X_test.shape[0])
    
    for n_fold, (trn_idx, val_idx) in enumerate(fold.split(sparse_train_all)):
        
        sparse_train = sparse_train_all[trn_idx]
        y_train = df_full.iloc[:ntrain]['TARGET'].iloc[trn_idx]
    
        sparse_val = sparse_train_all[val_idx]
        y_val = df_full.iloc[:ntrain]['TARGET'].iloc[val_idx]
    
        
        X_train_xgb = xgb.DMatrix(sparse_train, label=y_train)
        X_val_xgb = xgb.DMatrix(sparse_val, label=y_val)
        X_test = xgb.DMatrix(sparse_test)
        
        xgb_params = {
            'objective'       : 'binary:logistic',
            'eval_metric'     : 'auc',
            'seed'            : 42,
            'min_child_weight': 3,
            'max_depth'       : 6,
            'eta'             : 0.01,
            'colsample_bytree': 0.151,
            'subsample'       : 0.9,
        }
        
        print('='*30, ' Fold {} '.format(n_fold+1), '='*30)
        xgb_model = xgb.train(xgb_params
                              , dtrain=X_train_xgb
                              , evals=[(X_train_xgb, 'train'), (X_val_xgb, 'eval')]
                              , verbose_eval=500
                              , num_boost_round=10000
                              , early_stopping_rounds=200)
        
        oof_preds_xgb_1[val_idx] = xgb_model.predict(X_val_xgb)
        sub_preds_xgb_1 += xgb_model.predict(X_test) / fold.n_splits
        
with timer('Submission file'):
    sub['TARGET'] = sub_preds_xgb_1
    sub[['SK_ID_CURR', 'TARGET']].to_csv('sub_xgb_5f.csv', index=False)