
import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from kaggle_learn.utils import timer
from kaggle_learn.feature_engineering.statistics import *

with timer('Load train and test data'):
    train = pd.read_csv('application_train.csv')
    test = pd.read_csv('application_test.csv')
    ntrain = train.shape[0]
    id_col = 'SK_ID_CURR'
    target = 'TARGET'
    df_full = pd.concat([train, test])
    print(train.shape, test.shape)
    
with timer('Preprocessing'):
    df_full['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df_full['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    df_full = df_full.loc[df_full['CODE_GENDER'] != 'XNA']; ntrain = ntrain - 4
    
with timer('Get categorical features')
    features_category = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                         'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                         'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                         'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 
                         'HOUR_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']

    features_category_num = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                             'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                             'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                             'LIVE_CITY_NOT_WORK_CITY']

    features_category_num.extend(['FLAG_DOCUMENT_{}'.format(i) for i in range(2, 22)])
    features_category.extend(features_category_num)

with timer('Encoding categorical features'):
    for c in features_category:
        encoder = LabelEncoder()
        df_full[c] = encoder.fit_transform(df_full[c].astype(str))   

with timer('Join with bureau_processed'):
    bureau = pd.read_csv('bureau_processed.csv')
    print(df_full.shape)
    df_full = df_full.merge(bureau, on=['SK_ID_CURR'], how='left')
    print(df_full.shape)
    del bureau; gc.collect()
    
with timer('Join with pos_cash_balance_processed'):
    pos_cash_balance_processed = pd.read_csv('pos_cash_balance_processed.csv')
    df_full = df_full.merge(pos_cash_balance_processed, on=['SK_ID_CURR'], how='left')
    del pos_cash_balance_processed; gc.collect()
    print(df_full.shape)
    
with timer('Join with installments_payments_processed'):
    installments_payments_processed = pd.read_csv('installments_payments_processed.csv')
    df_full = df_full.merge(installments_payments_processed, on=['SK_ID_CURR'], how='left')
    del installments_payments_processed; gc.collect()
    print(df_full.shape)
    
with timer('Join with credit_card_balance_processed'):
    credit_card_balance_processed = pd.read_csv('credit_card_balance_processed.csv')
    print(credit_card_balance_processed['SK_ID_CURR'].nunique())
    df_full = df_full.merge(credit_card_balance_processed, on=['SK_ID_CURR'], how='left')
    del credit_card_balance_processed; gc.collect()
    print(df_full.shape)
    
with timer('Join with previous_application_processed'):
    previous_application_processed = pd.read_csv('previous_application_processed_2.csv')
    df_full = df_full.merge(previous_application_processed, on=['SK_ID_CURR'], how='left')
    del previous_application_processed; gc.collect()
    print(df_full.shape)
    
with timer('Simple feature engineering'):
    df_full['DOC_SUM'] = df_full[['FLAG_DOCUMENT_{}'.format(i) for i in range(2, 22)]].sum(axis=1)
    df_full['CREDIT_INCOME_RATIO'] = df_full['AMT_CREDIT'] / df_full['AMT_INCOME_TOTAL'] 
    df_full['ANNUITY_INCOMR_RATIO'] = df_full['AMT_ANNUITY'] / df_full['AMT_INCOME_TOTAL']
    df_full['ANNUITY_CREDIT_RATIO'] = df_full['AMT_ANNUITY'] / df_full['AMT_CREDIT'] 
    df_full['PRICE_CREDIT_RATIO'] = df_full['AMT_GOODS_PRICE'] / df_full['AMT_CREDIT']
    df_full['PRICE_INCOME_RATIO'] = df_full['AMT_GOODS_PRICE'] / df_full['AMT_INCOME_TOTAL']
    df_full['INCOME_FAM_AVG'] = df_full['AMT_INCOME_TOTAL'] / df_full['CNT_FAM_MEMBERS']
    df_full['ANNUITY_FAM_AVG'] = df_full['AMT_ANNUITY'] / df_full['CNT_FAM_MEMBERS']
    df_full['CREDIR_FAM_AVG'] = df_full['AMT_CREDIT'] / df_full['CNT_FAM_MEMBERS']
    df_full['CHILDREN_RATIO'] = df_full['CNT_CHILDREN'] / df_full['CNT_FAM_MEMBERS']
    df_full['EMPLOYED_RECENCY'] = df_full['DAYS_EMPLOYED'] / df_full['DAYS_BIRTH']
    df_full['ID_RECENCY'] = df_full['DAYS_ID_PUBLISH'] / df_full['DAYS_BIRTH']
    df_full['REGISTRATION_RECENCY'] = df_full['DAYS_REGISTRATION'] / df_full['DAYS_BIRTH']
    df_full['CAR_RECENCY'] = df_full['OWN_CAR_AGE'] / df_full['DAYS_BIRTH']
    df_full['CAR_EMPLOYED_RECENCY'] = df_full['OWN_CAR_AGE'] / df_full['DAYS_EMPLOYED']
    df_full['AGE'] = df_full['DAYS_BIRTH'] // 365
    df_full['INCOME_AGE_RATIO'] = df_full['AMT_INCOME_TOTAL'] / df_full['AGE'] 
    df_full['EMPLOYED_AGE'] = df_full['DAYS_EMPLOYED'] // 365
    df_full['INCOME_EMPLOYED_RATIO'] = df_full['AMT_INCOME_TOTAL'] / df_full['EMPLOYED_AGE']
    df_full['CNT_NON_CHILDREN'] = df_full['CNT_FAM_MEMBERS'] - df_full['CNT_CHILDREN']
    df_full['PRICE_ANNUITY_RATIO'] = df_full['AMT_GOODS_PRICE'] / df_full['AMT_ANNUITY']

with timer("Group by feature engineering"):
    agg_func_map = {
        "mean": add_group_mean,
        "median": add_group_median
    }
    
    agg_config = [
        (["ORGANIZATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"),
                                 ("AMT_INCOME_TOTAL", "median")]),
        (["OCCUPATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"),
                               ("AMT_INCOME_TOTAL", "median")]),
        (["ORGANIZATION_TYPE", "OCCUPATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"),
                                                    ("AMT_INCOME_TOTAL", "median")]),
        (["AGE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"), 
                                                           ("AMT_INCOME_TOTAL", "median"),
                                                           ("AMT_CREDIT", "mean"),
                                                           ("AMT_CREDIT", "median"),
                                                           ("AMT_ANNUITY", "mean"),
                                                           ("AMT_ANNUITY", "median")]),
        (["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"), 
                                                      ("AMT_INCOME_TOTAL", "median")])
    ]
    
    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            df_full = agg_func_map[agg_feat[1]](df_full, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
            df_full["{}_{}_diff".format(agg_feat[0], new_col)] = df_full[agg_feat[0]] - df_full[new_col]
            
with timer('Feature binning'):
    df_full['AGE'] = df_full['AGE'] // 5
    
with timer('Select features'):
    features = df_full.columns.tolist()
    features.remove(id_col)
    features.remove(target)

    features_rm = ['FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_21',
                   'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19',
                   'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_CONT_MOBILE',
                   
                   # bureau features to be removed
                   'BU_INTERBANK_CREDIT_CNT',
                   'BU_LOAN_FOR_PURCHASE_OF_SHARES_(MARGIN_LENDING)_CNT',
                   'BU_MOBILE_OPERATOR_LOAN_CNT',
                   'BU_CURRENCY_4_CNT',
                   'BU_BAD_DEBT_CNT',
                   'BU_BAD_DEBT_RATIO_STD',
                   'BU_SOLD_RATIO_STD',
                   'BU_CLOSED_RATIO_STD',
                   'BU_ACTIVE_RATIO_STD',

                   # credit card balance features to be removed
                   'CB_POS_DRAWING_BALANCE_RATIO_MIN',
                   'CB_POS_DRAWING_BALANCE_RATIO_MAX',
                   'CB_POS_DRAWING_BALANCE_RATIO_AVG',
                   'CB_BALANCE_LIMIT_RATIO_AVG',
                   'CB_BALANCE_LIMIT_RATIO_MIN',
                   'CB_BALANCE_LIMIT_RATIO_MAX', 
                   'CB_PAYMENT_AVG',
                   'CB_PAYMENT_MIN',
                   'CB_PAYMENT_MAX',
                   'CB_RECEIVABLE_TOTAL_DRAWING_RATIO_AVG', 
                   'CB_RECEIVABLE_TOTAL_DRAWING_RATIO_MIN',
                   'CB_RECEIVABLE_TOTAL_DRAWING_RATIO_MAX',
                   'CB_PREV_APP_CNT',
                   'CB_TOTAL_INSTALMENT',
                   'CB_INSTALMENTS_PER_LOAN',
                   'CB_INSTAL_MIN_REG_MISSED_AVG',
                   'CB_INSTAL_MIN_REG_MISSED_MIN',
                   'CB_CB_INSTAL_MIN_REG_PAYMENT_RATIO_AVG',
                   'CB_CB_INSTAL_MIN_REG_PAYMENT_RATIO_MIN',
                   'CB_CB_INSTAL_MIN_REG_PAYMENT_RATIO_MAX',
                   'CB_DRAWING_AVG_AVG',
                   'CB_DRAWING_AVG_MIN',
                   'CB_DRAWING_AVG_MAX',
                   
                   # previous application features to be removed
                   'PA_LAST_APPR_PA_APP_PRICE_RATIO',
                   'PA_PA_APP_PRICE_RATIO_AVG',
                   'PA_PA_APP_PRICE_RATIO_MIN',
                   'PA_PA_APP_PRICE_RATIO_MAX',
               
                   # pos cash balance features to be removed 
                   'PC_XNA_CNT',
                   'PC_SK_DPD_DEF_MIN',
                   'PC_SK_DPD_MIN',
                   'PC_LAST_5_SK_DPD_MIN', 
                   'PC_LAST_10_SK_DPD_MIN',
                   'PC_LAST_5_SK_DPD_DEF_MIN', 
                   'PC_LAST_10_SK_DPD_DEF_MIN']

    for c in features_rm:
        features.remove(c)
        if c in features_category:
            features_category.remove(c)
    print('Number of features = {}'.format(len(features)))

with open("df_full.pkl", "wb") as f:
    pickle.dump(df_full, f)

with open("features_category.pkl", "wb") as f:
    pickle.dump(features_category, f)
    
with open("features.pkl", "wb") as f:
    pickle.dump(features, f)