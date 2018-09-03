
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

with timer('Read data'):
    bureau = pd.read_csv('bureau.csv')
    bureau_balance = pd.read_csv('bureau_balance.csv')
    print(bureau.shape, bureau_balance.shape)
    
with timer('Join bureau_balance to bureau'):
    gp = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count().reset_index()
    gp.columns = ['SK_ID_BUREAU', 'MONTHS_BALANCE_COUNT']
    
    # MONTHS_BALANCE_COUNT: for each SK_ID_CURR, it corresponding to multiple SK_ID_BUREAU
    # each SK_ID_BUREAU has a series of MONTHS_BALANCE
    bureau = bureau.merge(gp, on=['SK_ID_BUREAU'], how='left')
    del gp; gc.collect()
    
    gp = add_group_value_count(bureau_balance, cols=['SK_ID_BUREAU', 'STATUS'], value='MONTHS_BALANCE', prefix='BUB_')\
                .drop(['MONTHS_BALANCE', 'STATUS'], axis=1)\
                .drop_duplicates(keep='first')
    bureau = bureau.merge(gp, on=['SK_ID_BUREAU'], how='left')
    del gp; gc.collect()
    
    bureau['BUB_0_RATIO'] = bureau['BUB_0_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_1_RATIO'] = bureau['BUB_1_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_2_RATIO'] = bureau['BUB_2_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_3_RATIO'] = bureau['BUB_3_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_4_RATIO'] = bureau['BUB_4_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_5_RATIO'] = bureau['BUB_5_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_C_RATIO'] = bureau['BUB_C_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    bureau['BUB_X_RATIO'] = bureau['BUB_X_CNT'] / bureau['MONTHS_BALANCE_COUNT']
    
with timer('Imputing missing values'):
    bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0.0, inplace=True)
    
with timer('Extract bureau simple features'):
    bureau = add_group_count(bureau, cols=['SK_ID_CURR'], cname='BU_PREV_CNT', value='SK_ID_BUREAU')
    bureau = add_group_value_count(bureau, cols=['SK_ID_CURR', 'CREDIT_CURRENCY'], value='SK_ID_BUREAU', prefix='BU_')
    bureau = add_group_value_count(bureau, cols=['SK_ID_CURR', 'CREDIT_ACTIVE'], value='SK_ID_BUREAU', prefix='BU_')
    bureau = add_group_value_count(bureau, cols=['SK_ID_CURR', 'CREDIT_TYPE'], value='SK_ID_BUREAU', prefix='BU_')
    
    bureau['BU_ACTIVE_RATIO'] = bureau['BU_ACTIVE_CNT'] / bureau['BU_PREV_CNT']
    bureau['BU_BAD_DEBT_RATIO'] = bureau['BU_BAD_DEBT_CNT'] / bureau['BU_PREV_CNT']
    bureau['BU_CLOSED_RATIO'] = bureau['BU_CLOSED_CNT'] / bureau['BU_PREV_CNT']
    bureau['BU_SOLD_RATIO'] = bureau['BU_SOLD_CNT'] / bureau['BU_PREV_CNT']
    
    bureau = add_group_nunique(bureau, cols=['SK_ID_CURR'], cname='BU_UNIQUE_CREDITY_TYPE', value='CREDIT_TYPE')
    bureau['BU_UNIQUE_CREDIT_TYPE_RATIO'] = bureau['BU_UNIQUE_CREDITY_TYPE'] / bureau['BU_PREV_CNT']
    
    gp = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(['SK_ID_CURR'])
    gp1 = gp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)
    gp1['DAYS_CREDIT'] = gp1['DAYS_CREDIT'] * (-1)
    gp1['DAYS_CREDIT_DIFF'] = gp1.groupby('SK_ID_CURR')['DAYS_CREDIT'].diff()
    gp1['DAYS_CREDIT_DIFF'] = gp1['DAYS_CREDIT_DIFF'].fillna(0.0)
    bureau = bureau.merge(gp1[['SK_ID_BUREAU', 'DAYS_CREDIT_DIFF']], on=['SK_ID_BUREAU'], how='left')
    
    bureau['DAYS_CREDIT_ENDDATE_BINARY'] = (bureau['DAYS_CREDIT_ENDDATE'] >= 0).astype(int)
    bureau = add_group_mean(bureau, cols=['SK_ID_CURR'], cname='BU_DAYS_CREDIT_ENDDATE_BINARY_AVG', value='DAYS_CREDIT_ENDDATE_BINARY')

    bureau_pos_enddate = bureau.loc[bureau['DAYS_CREDIT_ENDDATE_BINARY'] == 1]
    gp = bureau_pos_enddate[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE']].groupby(['SK_ID_CURR'])
    gp1 = gp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE'], ascending=True)).reset_index(drop=True)
    gp1['DAYS_CREDIT_ENDDATE_DIFF'] = gp1.groupby(['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].diff()
    gp1['DAYS_CREDIT_ENDDATE_DIFF'] = gp1['DAYS_CREDIT_ENDDATE_DIFF'].fillna(0.0)
    bureau = bureau.merge(gp1[['SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE_DIFF']], on=['SK_ID_BUREAU'], how='left')
    
with timer('Group by feature engineering'):
    agg_func_map = {
        "mean": add_group_mean,
        "max" : add_group_max,
        "min" : add_group_min,
        "sum" : add_group_sum
        "std" : add_group_std,        
    }
    
    agg_config = [
        (["SK_ID_CURR"], [("CREDIT_DAY_OVERDUR", "sum"),
                          ("CREDIT_DAY_OVERDUR", "mean"),
                          ("CREDIT_DAY_OVERDUR", "max"),
                          ("DAYS_CREDIT", "mean"),
                          ("DAYS_CREDIT", "max"),
                          ("DAYS_CREDIT", "min"),
                          ("DAYS_CREDIT_UPDATE", "mean"),
                          ("DAYS_CREDIT_UPDATE", "max"),
                          ("DAYS_CREDIT_UPDATE", "min"),
                          ("DAYS_ENDDATE_FACT", "min"),
                          ("DAYS_ENDDATE_FACT", "mean"),
                          ("DAYS_ENDDATE_FACT", "max"),
                          ("AMT_ANNUITY", "mean"),
                          ("AMT_ANNUITY", "max"),
                          ("AMT_ANNUITY", "min"),
                          ("AMT_ANNUITY", "std"),
                          ("AMT_CREDIT_SUM_OVERDUE", "sum"),
                          ("AMT_CREDIT_SUM_OVERDUE", "max"),
                          ("AMT_CREDIT_SUM_OVERDUE", "mean"),
                          ("AMT_CREDIT_MAX_OVERDUE", "max"),
                          ("AMT_CREDIT_MAX_OVERDUE", "min"),
                          ("CNT_CREDIT_PROLONG", "sum"),
                          ("CNT_CREDIT_PROLONG", "mean"),
                          ("CNT_CREDIT_PROLONG", "max"),
                          ("AMT_CREDIT_SUM", "sum"),
                          ("AMT_CREDIT_SUM", "mean"),
                          ("AMT_CREDIT_SUM", "max"),
                          ("AMT_CREDIT_SUM", "min"),
                          ("AMT_CREDIT_SUM_DEBT", "sum"),
                          ("AMT_CREDIT_SUM_DEBT", "mean"),
                          ("AMT_CREDIT_SUM_DEBT", "max"),
                          ("AMT_CREDIT_SUM_LIMIT", "max"),
                          ("AMT_CREDIT_SUM_LIMIT", "mean"),
                          ("MONTHS_BALANCE_COUNT", "mean"),
                          ("MONTHS_BALANCE_COUNT", "sum"),
                          ("MONTHS_BALANCE_COUNT", "max"),
                          ("MONTHS_BALANCE_COUNT", "min"),
                          ("BUB_0_CNT", "mean"),
                          ("BUB_0_CNT", "sum"),
                          ("BUB_0_CNT", "max"),
                          ("BUB_0_CNT", "min"),
                          ("BUB_1_CNT", "mean"),
                          ("BUB_1_CNT", "sum"),
                          ("BUB_1_CNT", "max"),
                          ("BUB_1_CNT", "min"),
                          ("BUB_2_CNT", "mean"),
                          ("BUB_2_CNT", "sum"),
                          ("BUB_2_CNT", "max"),
                          ("BUB_2_CNT", "min"),
                          ("BUB_3_CNT", "mean"),
                          ("BUB_3_CNT", "sum"),
                          ("BUB_3_CNT", "max"),
                          ("BUB_3_CNT", "min"),
                          ("BUB_4_CNT", "mean"),
                          ("BUB_4_CNT", "sum"),
                          ("BUB_4_CNT", "max"),
                          ("BUB_4_CNT", "min"),
                          ("BUB_5_CNT", "mean"),
                          ("BUB_5_CNT", "sum"),
                          ("BUB_5_CNT", "max"),
                          ("BUB_5_CNT", "min"),
                          ("BUB_C_CNT", "mean"),
                          ("BUB_C_CNT", "sum"),
                          ("BUB_C_CNT", "max"),
                          ("BUB_C_CNT", "min"),
                          ("BUB_X_CNT", "mean"),
                          ("BUB_X_CNT", "sum"),
                          ("BUB_X_CNT", "max"),
                          ("BUB_X_CNT", "min"),
                          ("BUB_0_CNT_RATIO", "mean"),
                          ("BUB_0_CNT_RATIO", "sum"),
                          ("BUB_0_CNT_RATIO", "max"),
                          ("BUB_0_CNT_RATIO", "min"),
                          ("BUB_1_CN_RATIOT", "mean"),
                          ("BUB_1_CNT_RATIO", "sum"),
                          ("BUB_1_CNT_RATIO", "max"),
                          ("BUB_1_CNT_RATIO", "min"),
                          ("BUB_2_CNT_RATIO", "mean"),
                          ("BUB_2_CNT_RATIO", "sum"),
                          ("BUB_2_CNT_RATIO", "max"),
                          ("BUB_2_CNT_RATIO", "min"),
                          ("BUB_3_CNT_RATIO", "mean"),
                          ("BUB_3_CNT_RATIO", "sum"),
                          ("BUB_3_CNT_RATIO", "max"),
                          ("BUB_3_CNT_RATIO", "min"),
                          ("BUB_4_CNT_RATIO", "mean"),
                          ("BUB_4_CNT_RATIO", "sum"),
                          ("BUB_4_CNT_RATIO", "max"),
                          ("BUB_4_CNT_RATIO", "min"),
                          ("BUB_5_CNT_RATIO", "mean"),
                          ("BUB_5_CNT_RATIO", "sum"),
                          ("BUB_5_CNT_RATIO", "max"),
                          ("BUB_5_CNT_RATIO", "min"),
                          ("BUB_C_CNT_RATIO", "mean"),
                          ("BUB_C_CNT_RATIO", "sum"),
                          ("BUB_C_CNT_RATIO", "max"),
                          ("BUB_C_CNT_RATIO", "min"),
                          ("BUB_X_CNT_RATIO", "mean"),
                          ("BUB_X_CNT_RATIO", "sum"),
                          ("BUB_X_CNT_RATIO", "max"),
                          ("BUB_X_CNT_RATIO", "min"),
                          ("BUB_ACTIVE_CNT_RATIO", "mean"),
                          ("BUB_ACTIVE_CNT_RATIO", "sum"),
                          ("BUB_ACTIVE_CNT_RATIO", "max"),
                          ("BUB_ACTIVE_CNT_RATIO", "min"),
                          ("BUB_CLOSED_CNT_RATIO", "mean"),
                          ("BUB_CLOSED_CNT_RATIO", "sum"),
                          ("BUB_CLOSED_CNT_RATIO", "max"),
                          ("BUB_CLOSED_CNT_RATIO", "min"),
                          ("BUB_BAD_DEBT_CNT_RATIO", "mean"),
                          ("BUB_BAD_DEBT_CNT_RATIO", "sum"),
                          ("BUB_BAD_DEBT_CNT_RATIO", "max"),
                          ("BUB_BAD_DEBT_CNT_RATIO", "min"),
                          ("BUB_SOLD_CNT_RATIO", "mean"),
                          ("BUB_SOLD_CNT_RATIO", "sum"),
                          ("BUB_SOLD_CNT_RATIO", "max"),
                          ("BUB_SOLD_CNT_RATIO", "min"),
                          ("DAYS_CREDIT_DIFF", "mean"),
                          ("DAYS_CREDIT_DIFF", "max"),
                          ("DAYS_CREDIT_DIFF", "std"),
                          ("DAYS_CREDIT_ENDDATE_DIFF", "mean"),
                          ("DAYS_CREDIT_ENDDATE_DIFF", "max"),
                          ("DAYS_CREDIT_ENDDATE_DIFF", "std")])
    ]
    
    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            bureau = agg_func_map[agg_feat[1]](bureau, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
            
    bureau['BU_AMT_CREDIT_OVERDUR_DEBT_RATIO'] = bureau['BU_AMT_CREDIT_SUM_OVERDUE_SUM'] / bureau['BU_AMT_CREDIT_SUM_DEBT_SUM']
    bureau['BU_AMT_CREDIT_DEBT_CREDIT_RATIO'] = bureau['BU_AMT_CREDIT_SUM_DEBT_SUM'] / bureau['BU_AMT_CREDIT_SUM_SUM']

with timer("Generating processed df"):
    exclude_features_bureau = ['SK_ID_BUREAU', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE',
                               'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
                               'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                               'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
                               'AMT_ANNUITY', 'MONTHS_BALANCE_COUNT', 
                               'BUB_0_CNT', 'BUB_1_CNT', 'BUB_2_CNT', 'BUB_3_CNT',
                               'BUB_4_CNT', 'BUB_5_CNT', 'BUB_C_CNT', 'BUB_X_CNT',
                               'BUB_0_RATIO', 'BUB_1_RATIO', 'BUB_2_RATIO', 'BUB_3_RATIO',
                               'BUB_4_RATIO', 'BUB_5_RATIO', 'BUB_C_RATIO', 'BUB_X_RATIO',
                               'BU_ACTIVE_RATIO', 'BU_BAD_DEBT_RATIO', 'BU_CLOSED_RATIO', 'BU_SOLD_RATIO',
                               'DAYS_CREDIT_DIFF', 'DAYS_CREDIT_ENDDATE_DIFF', 'DAYS_CREDIT_ENDDATE_BINARY',]
    
    bureau_processed = bureau[[c for c in bureau.columns if c not in exclude_features_bureau]].drop_duplicates(keep='first')
    print('bureau_processed shape = {}'.format(bureau_processed.shape))
    bureau_processed.to_csv('bureau_processed.csv', index=False)