
import os
import gc
import pickle
import datetime
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from kaggle_learn.utils import timer
from kaggle_learn.metrics import rmse

# fork from https://www.kaggle.com/ogrellier/feature-selection-with-null-importances/notebook
def get_feature_importances(X_train, y_train, shuffle=True, seed=None):
    if shuffle:
        y_train = pd.DataFrame(y_train, columns=['target']).copy().sample(frac=1.0)

    if isinstance(y_train, pd.DataFrame):
        X_train_lgb = lgb.Dataset(X_train.values, y_train.values.reshape(-1), free_raw_data=False, silent=True)
    else:
        X_train_lgb = lgb.Dataset(X_train.values, y_train.reshape(-1), free_raw_data=False, silent=True)

    lgb_params = {
        'objective'         : 'regression',
        'boosting_type'     : 'gbdt',
        'metric'            : 'rmse',
        'learning_rate'     : 0.01,
        'max_depth'         : 8,
        'num_leaves'        : 120,
        'min_data_in_leaf'  : 90,
        'feature_fraction'  : 0.185,
        'bagging_fraction'  : 1,
        'data_random_seed'  : seed,
        'lambda_l1'         : 0.4,
        'lambda_l2'         : 0.4,
        'cat_l2'            : 15,
        'min_gain_to_split' : 0.00,
        'min_data_per_group': 100,
        'max_bin'           : 255,
        'nthread'           : 4
    }

    lgb_regressor = lgb.train(params=lgb_params, train_set=X_train_lgb, num_boost_round=1000)
    imp_df = pd.DataFrame()
    imp_df['feature'] = list(X_train.columns)
    imp_df['importance_gain'] = lgb_regressor.feature_importance(importance_type='gain')
    imp_df['importance_split'] = lgb_regressor.feature_importance(importance_type='split')
    return imp_df


with timer('Load data'):
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    hist_transac_info = pd.read_csv('hist_transac_info.csv')
    hist_transac_amount = pd.read_csv('hist_transac_amount.csv')
    hist_transac_time = pd.read_csv('hist_transac_time.csv')
    hist_transac_info_a = pd.read_csv('hist_transac_info_a.csv')
    hist_transac_amount_a = pd.read_csv('hist_transac_amount_a.csv')
    hist_transac_time_a = pd.read_csv('hist_transac_time_a.csv')
    hist_transac_merchant_lda_comp = pd.read_csv('hist_transac_merchant_category_lda_comp_0.csv')
    hist_transac_merchant_lda_comp_2 = pd.read_csv('hist_transac_merchant_category_lda_comp_2.csv')
    hist_transac_merchantid_lda_comp = pd.read_csv('hist_transac_merchant_id_lda_comp_0_1.csv')
    print(hist_transac_info.shape, hist_transac_time.shape, hist_transac_amount.shape)
    print(hist_transac_info_a.shape, hist_transac_time_a.shape, hist_transac_amount_a.shape)
    print(hist_transac_merchant_lda_comp.shape, hist_transac_merchant_lda_comp_2.shape)

    hist_feats = hist_transac_info.merge(hist_transac_amount, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_time, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_info_a, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_amount_a, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_time_a, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_merchant_lda_comp, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_merchant_lda_comp_2, on='card_id', how='left')
    hist_feats = hist_feats.merge(hist_transac_merchantid_lda_comp, on='card_id', how='left')

    del hist_transac_info, hist_transac_amount, hist_transac_time
    del hist_transac_info_a, hist_transac_amount_a, hist_transac_time_a
    del hist_transac_merchant_lda_comp, hist_transac_merchant_lda_comp_2
    gc.collect()

    new_transac_info = pd.read_csv('new_transac_info.csv')
    new_transac_amount = pd.read_csv('new_transac_amount.csv')
    new_transac_time = pd.read_csv('new_transac_time.csv')
    print(new_transac_info.shape, new_transac_time.shape, new_transac_amount.shape)

    new_feats = new_transac_info.merge(new_transac_amount, on='card_id', how='left')
    new_feats = new_feats.merge(new_transac_time, on='card_id', how='left')
    del new_transac_info, new_transac_amount, new_transac_time
    gc.collect()

    merchant_repurchase_feats = pd.read_csv('merchant_repurchase_rates.csv')
    print(train.shape, test.shape, hist_feats.shape, new_feats.shape, merchant_repurchase_feats.shape)

with timer('Outlier processing'):
    is_outlier = train['target'] < -20.
    outlier_idx = train.index[train['target'] < -20.]
    normal_idx = train.index[train['target'] >= -20.]
    print('Number of outliers = {}, normal = {}'.format(len(outlier_idx), len(normal_idx)))

with timer('Join data'):

    train = train.merge(hist_feats, on=['card_id'], how='left')
    test = test.merge(hist_feats, on=['card_id'], how='left')
    print(train.shape, test.shape)

    train = train.merge(new_feats, on=['card_id'], how='left')
    test = test.merge(new_feats, on=['card_id'], how='left')
    print(train.shape, test.shape)

    train = train.merge(merchant_repurchase_feats, on=['card_id'], how='left')
    test = test.merge(merchant_repurchase_feats, on=['card_id'], how='left')
    print(train.shape, test.shape)

with timer('More features'):
    for df in [train, test]:
        # time related
        df['first_active_year'] = df['first_active_month'].astype(str).apply(lambda x: x[:4])
        df['first_active_mon'] = df['first_active_month'].astype(str).apply(lambda x: x[-2:])
        df['ref_first_month_diff_days'] = (pd.to_datetime(df['reference_month']) - pd.to_datetime(df['first_active_month'])).dt.days.values
        df['ref_elapsed_time_days'] = (pd.to_datetime('2018-12-31') - pd.to_datetime(df['reference_month'])).dt.days.values

        df['hist_purchase_active_diff'] = (pd.to_datetime(df['hist_purchase_date_first'].astype(str).apply(lambda x: x[:7])) - pd.to_datetime(df['first_active_month'])).dt.days.values
        df['hist_purchase_recency'] = (pd.to_datetime('2018-12-31') - pd.to_datetime(df['hist_purchase_date_last'])).dt.days.values
        df['new_purchase_recency'] = (pd.to_datetime('2018-12-31') - pd.to_datetime(df['new_purchase_date_last'])).dt.days.values

        df['new_purchase_date_last'] = pd.to_datetime(df['new_purchase_date_last']).astype(np.int64) * 1e-9
        df['new_purchase_date_first'] = pd.to_datetime(df['new_purchase_date_first']).astype(np.int64) * 1e-9
        df['hist_purchase_date_last'] = pd.to_datetime(df['hist_purchase_date_last']).astype(np.int64) * 1e-9
        df['hist_purchase_date_first'] = pd.to_datetime(df['hist_purchase_date_first']).astype(np.int64) * 1e-9

        # historical / new transaction interaction
        df['new_hist_transac_amount_sum_ratio'] = df['new_transac_amount_sum'].values / df['hist_transac_amount_sum'].values
        df['new_hist_transac_amount_max_ratio'] = df['new_transac_amount_max'].values / df['hist_transac_amount_max'].values
        df['hist_transac_amount_month_mean'] = df['hist_transac_amount_sum'].values / df['hist_transac_monthlag_min'].values
        df['hist_transac_count_month_mean'] = df['hist_transac_count'].values / df['hist_transac_monthlag_min'].values

        df['new_hist_transac_a_amount_sum_ratio'] = df['new_transac_amount_sum'].values / df['hist_transac_a_amount_sum'].values
        df['new_hist_transac_a_amount_max_ratio'] = df['new_transac_amount_max'].values / df['hist_transac_a_amount_max'].values
        df['hist_transac_a_amount_month_mean'] = df['hist_transac_a_amount_sum'].values / df['hist_transac_a_monthlag_min'].values
        df['hist_transac_a_count_month_mean'] = df['hist_transac_approved_count'].values / df['hist_transac_a_monthlag_min'].values

        df['transac_month_lag_0_1_ratio'] = df['hist_transac_month_lag=0_count'].values / (1. + df['new_transac_month_lag=1_count'].values)
        df['transac_a_month_lag_0_1_ratio'] = df['hist_transac_a_month_lag=0_count'].values / (1. + df['new_transac_month_lag=1_count'].values)
        df['transac_month_lag_0_2_ratio'] = df['hist_transac_month_lag=0_count'].values / (1. + df['new_transac_month_lag=2_count'].values)
        df['transac_a_month_lag_0_2_ratio'] = df['hist_transac_a_month_lag=0_count'].values / (1. + df['new_transac_month_lag=2_count'].values)
        df['transac_month_lag_2_-2_ratio'] = df['hist_transac_monthlag_last_2_amount'].values / df['new_transac_amount_sum'].values

        df['hist_new_transac_count_ratio'] = df['new_transac_count'].values / (1. + df['hist_transac_approved_count'].values)

        df['hist_clv'] = df['hist_transac_count'].values * df['hist_transac_amount_sum'].values / df['hist_month_diff_mean'].values
        df['new_clv'] = df['new_transac_count'].values * df['new_transac_amount_sum'].values / df['new_month_diff_mean'].values
        df['clv_ratio'] = df['new_clv'].values / df['hist_clv'].values
        
        df['installment_total_sum'] = df['hist_transac_installments_sum'].values + df['new_transac_installments_sum'].values
        
        for i in range(1, 7):
            for j in range(1, 3):
                df['new_hist_purchase_amount_ratio_{}_{}'.format(i, j)] = df['new_transac_monthlag_last_{}_amount'.format(j)] / df['hist_transac_monthlag_last_{}_amount'.format(i)]
                df['new_hist_purchase_amount_log_ratio_{}_{}'.format(i, j)] = np.log2(df['new_hist_purchase_amount_ratio_{}_{}'.format(i, j)])
                df['new_hist_a_purchase_amount_ratio_{}_{}'.format(i, j)] = df['new_transac_monthlag_last_{}_amount'.format(j)] / df['hist_transac_a_monthlag_last_{}_amount'.format(i)]
                df['new_hist_a_purchase_amount_log_ratio_{}_{}'.format(i, j)] = np.log2(df['new_hist_a_purchase_amount_ratio_{}_{}'.format(i, j)])
        
        df['new_hist_transac_amount_sum_log_ratio'] = np.log2(df['new_hist_transac_amount_sum_ratio'])
        df['new_hist_transac_amount_max_log_ratio'] = np.log2(df['new_hist_transac_amount_max_ratio'])
        df['new_hist_transac_a_amount_sum_log_ratio'] = np.log2(df['new_hist_transac_amount_sum_ratio'])
        df['new_hist_transac_a_amount_max_log_ratio'] = np.log2(df['new_hist_transac_amount_max_ratio'])

with timer('Encoding categorical features'):
    lbl_encoder = LabelEncoder()
    for c in ['first_active_year', 'first_active_mon']:
        train[c] = lbl_encoder.fit_transform(train[c].astype(str))
        test[c] = lbl_encoder.fit_transform(test[c].astype(str))

with timer('Set features'):
    try:
        drop_feats = ['hist_transac_amount_diff_ratio', 'transac_amount_global_mean',
                      'hist_transac_denied_approved_amount_ratio']
        train.drop(drop_feats, axis=1, inplace=True)
        test.drop(drop_feats, axis=1, inplace=True)
    except:
        print('No features to be removed')

    features_full = train.columns.tolist()
    features_categorical = ['feature_1', 'feature_2', 'feature_3',
                            'first_active_year', 'first_active_mon']
    target = 'target'
    for c in ['target', 'card_id', 'first_active_month', 'reference_month', 'ref_first_month_diff_days']:
        features_full.remove(c)
        if c in features_categorical:
            features_categorical.remove(c)

    print('Number of features = {}'.format(len(features_full)))

with timer('Prepare for LightGBM (feature importance)'):
    seed = 4590
    num_folds = 10
    cv = True
    stratified = True
    remove_outliers = False
    if remove_outliers:
        X_train_all = train.iloc[normal_idx][features_full]
        y_train_all = train.iloc[normal_idx][target]
    else:
        X_train_all = train[features_full]
        y_train_all = train[target]

    X_test = test[features_full]
    sub = pd.read_csv('sample_submission.csv')

with timer('Get actual importance df'):
    actual_imp_df = get_feature_importances(X_train_all, y_train_all.values, shuffle=False)

with timer('Get null importance df'):
    null_imp_df = pd.DataFrame()
    nb_runs = 80

    start = time.time()
    dsp = ''

    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(X_train_all, y_train_all.values, shuffle=True)
        imp_df['run'] = i + 1

        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)

        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = '\nDone with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)

with timer('Get scores'):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

with timer('Prepare and run LightGBM / Submission'):
    seed = 4590
    num_folds = 10
    cv = True
    stratified = True
    remove_outliers = False
    sub = pd.read_csv('sample_submission.csv')

    # selected features
    features_selected_1 = scores_df.loc[scores_df['split_score'] > 0.00]['feature'].tolist()
    features_selected_2 = scores_df.loc[scores_df['gain_score'] > 1.00]['feature'].tolist()
    features_selected_3 = list(set(features_selected_1).union(set(features_selected_2)))
    features_selected_4 = list(set(features_selected_1).intersection(set(features_selected_2)))
    
    features_selected = list(set(features_selected_4) - set(features_selected_3))
    intersect_features_num = len(features_selected_3)
    np.random.seed(seed)
    select_idx = np.random.choice(intersect_features_num, int(intersect_features_num * 0.8), replace=False)
    for idx in select_idx:
        features_selected.append(features_selected_3[idx])
    features_selected_5 = features_selected.copy()

    features_set = [features_selected_1, features_selected_2, features_selected_3, features_selected_4, features_selected_5]

    for f_idx, features_selected in enumerate(features_set):
        print('Selected features = {}'.format(len(features_selected)))
        if remove_outliers:
            X_train_all = train.iloc[normal_idx][features_selected]
            y_train_all = train.iloc[normal_idx][target]
        else:
            X_train_all = train[features_selected]
            y_train_all = train[target]

        X_test = test[features_selected]

        for c in features_categorical:
            if c not in features_selected:
                features_categorical.remove(c)
        
        # currently use k-folds cross validation
        if cv:
            if stratified:
                fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            else:
                fold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            model_list = []
            oof_preds_lgb = np.zeros(X_train_all.shape[0])
            sub_preds_lgb = np.zeros(X_test.shape[0])

            for n_fold, (trn_idx, val_idx) in enumerate(fold.split(X_train_all, is_outlier)):
                print('\tFold {}'.format(n_fold + 1))
                X_train = X_train_all.iloc[trn_idx]
                y_train = y_train_all.iloc[trn_idx]

                X_val = X_train_all.iloc[val_idx]
                y_val = y_train_all.iloc[val_idx]

                X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=features_selected, categorical_feature=features_categorical)
                X_val_lgb = lgb.Dataset(X_val, y_val, feature_name=features_selected, categorical_feature=features_categorical)

                # lightgbm hyper-parameters: could be different for different feature sets
                # that what I exactly did for my final submission
                lgb_params = {
                    'objective'         : 'regression',
                    'boosting_type'     : 'gbdt',
                    'metric'            : 'rmse',
                    'learning_rate'     : 0.01,
                    'max_depth'         : 8,
                    'num_leaves'        : 120,
                    'min_data_in_leaf'  : 90,
                    'feature_fraction'  : 0.185,
                    'bagging_fraction'  : 1,
                    'data_random_seed'  : seed,
                    'lambda_l1'         : 0.4,
                    'lambda_l2'         : 0.4,
                    'cat_l2'            : 15,
                    'min_gain_to_split' : 0.00,
                    'min_data_per_group': 100,
                    'max_bin'           : 255,
                    'nthread'           : 4
                }

                lgb_model = lgb.train(lgb_params
                                      , train_set=X_train_lgb
                                      , valid_sets=[X_train_lgb, X_val_lgb]
                                      , num_boost_round=3000
                                      , early_stopping_rounds=200
                                      , verbose_eval=400)
                model_list.append(lgb_model)
                oof_preds_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
                sub_preds_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / fold.n_splits
        
        print('CV RMSE = {:.6f}'.format(rmse(y_train_all.values, oof_preds_lgb)))
        sub['target'] = sub_preds_lgb
        sub.to_csv('sub_lgb_feats_select_v{}.csv'.format(f_idx), index=False)