
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
    credit_card_balance = pd.read_csv('credit_card_balance.csv')
    print(credit_card_balance.shape)

with timer('Simple features'):
    credit_card_balance['CB_BALANCE_LIMIT_RATIO'] = credit_card_balance['AMT_BALANCE'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['CB_ATM_DRAWING_BALANCE_RATIO'] = credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'] / credit_card_balance['AMT_BALANCE']
    credit_card_balance['CB_ATM_DRAWING_LIMIT_RATIO'] = credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['CB_DRAWING_BALANCE_RATIO'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] / credit_card_balance['AMT_BALANCE']
    credit_card_balance['CB_DRAWING_LIMIT_RATIO'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['CB_POS_DRAWING_BALANCE_RATIO'] = credit_card_balance['AMT_DRAWINGS_POS_CURRENT'] / credit_card_balance['AMT_BALANCE']
    credit_card_balance['CB_POS_DRAWING_LIMIT_RATIO'] = credit_card_balance['AMT_DRAWINGS_POS_CURRENT'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['CB_DRAWING_ATM_RATIO'] = credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'] / credit_card_balance['AMT_DRAWINGS_CURRENT']
    credit_card_balance['CB_DRAWING_POS_RATIO'] = credit_card_balance['AMT_DRAWINGS_POS_CURRENT'] / credit_card_balance['AMT_DRAWINGS_CURRENT']
    credit_card_balance['CB_MIN_INSTALLMENT_BALANCE_RATIO'] = credit_card_balance['AMT_INST_MIN_REGULARITY'] / credit_card_balance['AMT_BALANCE']
    credit_card_balance['CB_MIN_INSTALLMENT_DRAWING_RAITO'] = credit_card_balance['AMT_INST_MIN_REGULARITY'] / credit_card_balance['AMT_DRAWINGS_CURRENT']
    credit_card_balance['CB_PAYMENT_BALANCE_RATIO'] = credit_card_balance['AMT_PAYMENT_TOTAL_CURRENT'] / credit_card_balance['AMT_BALANCE']
    credit_card_balance['CB_PAYMENT_LIMIT_RATIO'] = credit_card_balance['AMT_PAYMENT_TOTAL_CURRENT'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['CB_RECEIVABLE_PRINCIPAL_TOTAL_RATIO'] = credit_card_balance['AMT_RECEIVABLE_PRINCIPAL'] / credit_card_balance['AMT_TOTAL_RECEIVABLE']
    credit_card_balance['CB_RECEIVABLE_TOTAL_DRAWING_RATIO'] = credit_card_balance['AMT_TOTAL_RECEIVABLE'] / credit_card_balance['AMT_DRAWINGS_CURRENT']
    credit_card_balance['CB_'+'INSTAL_MIN_REG_CNT_INSTAL_MATURE_'+ 'RATIO'] = credit_card_balance['AMT_INST_MIN_REGULARITY'] / credit_card_balance['CNT_INSTALMENT_MATURE_CUM']
    credit_card_balance['CB_'+'INSTAL_MIN_REG_LIMIT_ACTUAL_'+ 'RATIO'] = credit_card_balance['AMT_INST_MIN_REGULARITY'] / credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance = add_group_count(credit_card_balance, cols=['SK_ID_CURR'], cname='CB_PREV_APP_CNT', value='SK_ID_PREV')
    gp = credit_card_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index()
    gp1 = gp.groupby(['SK_ID_CURR'])['CNT_INSTALMENT_MATURE_CUM'].sum()\
            .reset_index()\
            .rename(index=str, columns={'CNT_INSTALMENT_MATURE_CUM': 'CB_TOTAL_INSTALMENT'})
    
    credit_card_balance = credit_card_balance.merge(gp1[['SK_ID_CURR', 'CB_TOTAL_INSTALMENT']], on=['SK_ID_CURR'], how='left')
    credit_card_balance['CB_INSTALMENTS_PER_LOAN'] = credit_card_balance['CB_TOTAL_INSTALMENT'] / credit_card_balance['CB_PREV_APP_CNT']
    credit_card_balance['CB_' + 'INSTAL_MIN_REG_PAYMENT_' + 'RATIO'] = credit_card_balance['AMT_INST_MIN_REGULARITY'] / credit_card_balance['AMT_PAYMENT_TOTAL_CURRENT']
    credit_card_balance['CB_INSTAL_MIN_REG_MISSED'] = (credit_card_balance['CB_INSTAL_MIN_REG_PAYMENT_RATIO'] < 1).astype(int)
    credit_card_balance['CB_DRAWING_AVG'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] / credit_card_balance['CNT_DRAWINGS_CURRENT']

with timer('Group by feature engineering'):
    agg_func_map = {
        "mean": add_group_mean,
        "max" : add_group_max,
        "min" : add_group_min,
        "sum" : add_group_sum
        "std" : add_group_std,        
    }
    
    agg_config = [
        (["SK_ID_CURR"], [("CNT_DRAWINGS_CURRENT", "sum"),
                          ("CNT_DRAWINGS_CURRENT", "mean"),
                          ("CNT_DRAWINGS_CURRENT", "max"),
                          ("CNT_DRAWINGS_CURRENT", "min"),
                          ("CNT_INSTALMENT_MATURE_CUM", "sum"),
                          ("CNT_INSTALMENT_MATURE_CUM", "mean"),
                          ("CNT_INSTALMENT_MATURE_CUM", "max"),
                          ("CNT_INSTALMENT_MATURE_CUM", "min"),
                          ("AMT_BALANCE", "mean"),
                          ("AMT_BALANCE", "max"),
                          ("AMT_BALANCE", "min"),
                          ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
                          ("AMT_CREDIT_LIMIT_ACTUAL", "max"),
                          ("AMT_CREDIT_LIMIT_ACTUAL", "min"),
                          ("AMT_INST_MIN_REGULARITY", "mean"),
                          ("AMT_INST_MIN_REGULARITY", "max"),
                          ("AMT_INST_MIN_REGULARITY", "min"),
                          ("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
                          ("AMT_PAYMENT_TOTAL_CURRENT", "max"),
                          ("AMT_PAYMENT_TOTAL_CURRENT", "min"),
                          ("CB_BALANCE_LIMIT_RATIO", "mean"),
                          ("CB_BALANCE_LIMIT_RATIO", "max"),
                          ("CB_BALANCE_LIMIT_RATIO", "min"),
                          ("CB_ATM_DRAWING_BALANCE_RATIO", "mean"),
                          ("CB_ATM_DRAWING_BALANCE_RATIO", "min"),
                          ("CB_ATM_DRAWING_BALANCE_RATIO", "max"),
                          ("CB_ATM_DRAWING_LIMIT_RATIO", "mean"),
                          ("CB_ATM_DRAWING_LIMIT_RATIO", "min"),
                          ("CB_ATM_DRAWING_LIMIT_RATIO", "max"),
                          ("CB_DRAWING_BALANCE_RATIO", "mean"),
                          ("CB_DRAWING_BALANCE_RATIO", "min"),
                          ("CB_DRAWING_BALANCE_RATIO", "max"),
                          ("CB_DRAWING_LIMIT_RATIO", "mean"),
                          ("CB_DRAWING_LIMIT_RATIO", "min"),
                          ("CB_DRAWING_LIMIT_RATIO", "max"),
                          ("CB_POS_DRAWING_BALANCE_RATIO", "mean"),
                          ("CB_POS_DRAWING_BALANCE_RATIO", "min"),
                          ("CB_POS_DRAWING_BALANCE_RATIO", "max"),
                          ("CB_POS_DRAWING_LIMIT_RATIO", "mean"),
                          ("CB_POS_DRAWING_LIMIT_RATIO", "min"),
                          ("CB_POS_DRAWING_LIMIT_RATIO", "max"),
                          ("CB_MIN_INSTALLMENT_BALANCE_RATIO", "mean"),
                          ("CB_MIN_INSTALLMENT_BALANCE_RATIO", "min"),
                          ("CB_MIN_INSTALLMENT_BALANCE_RATIO", "max"),
                          ("CB_MIN_INSTALLMENT_DRAWING_RAITO", "mean"),
                          ("CB_MIN_INSTALLMENT_DRAWING_RAITO", "mean"),
                          ("CB_MIN_INSTALLMENT_DRAWING_RAITO", "min"),
                          ("CB_MIN_INSTALLMENT_DRAWING_RAITO", "max"),
                          ("CB_PAYMENT_BALANCE_RATIO", "mean"),
                          ("CB_PAYMENT_BALANCE_RATIO", "min"),
                          ("CB_PAYMENT_BALANCE_RATIO", "max"),
                          ("CB_PAYMENT_LIMIT_RATIO", "mean"),
                          ("CB_PAYMENT_LIMIT_RATIO", "min"),
                          ("CB_PAYMENT_LIMIT_RATIO", "max"),
                          ("CB_RECEIVABLE_TOTAL_DRAWING_RATIO", "mean"),
                          ("CB_RECEIVABLE_TOTAL_DRAWING_RATIO", "min"),
                          ("CB_RECEIVABLE_TOTAL_DRAWING_RATIO", "max"),
                          ("CB_SK_DPD", "sum"),
                          ("CB_SK_DPD", "mean"),
                          ("CB_SK_DPD", "max"),
                          ("CB_SK_DPD_DEF", "sum"),
                          ("CB_SK_DPD_DEF", "mean"),
                          ("CB_SK_DPD_DEF", "max"),
                          ("CB_INSTAL_MIN_REG_CNT_INSTAL_MATURE_RATIO", "mean"),
                          ("CB_INSTAL_MIN_REG_CNT_INSTAL_MATURE_RATIO", "min"),
                          ("CB_INSTAL_MIN_REG_CNT_INSTAL_MATURE_RATIO", "max"),
                          ("CB_INSTAL_MIN_REG_LIMIT_ACTUAL_RATIO", "mean"),
                          ("CB_INSTAL_MIN_REG_LIMIT_ACTUAL_RATIO", "min"),
                          ("CB_INSTAL_MIN_REG_LIMIT_ACTUAL_RATIO", "max"),
                          ("CB_INSTAL_MIN_REG_MISSED", "mean"),
                          ("CB_INSTAL_MIN_REG_MISSED", "min"),
                          ("CB_INSTAL_MIN_REG_PAYMENT_RATIO", "mean"),
                          ("CB_INSTAL_MIN_REG_PAYMENT_RATIO", "min"),
                          ("CB_INSTAL_MIN_REG_PAYMENT_RATIO", "max"),
                          ("CB_DRAWING_AVG", "mean"),
                          ("CB_DRAWING_AVG", "min"),
                          ("CB_DRAWING_AVG", "max")])
    ]
    
    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            credit_card_balance = agg_func_map[agg_feat[1]](credit_card_balance, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
    
with timer('Generate credit_card_balance_processed'):
    credit_card_balance_processed = credit_card_balance.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE',
                                                              'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
                                                              'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                                                              'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
                                                              'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
                                                              'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
                                                              'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
                                                              'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
                                                              'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS', 'SK_DPD',
                                                              'SK_DPD_DEF', 'CB_BALANCE_LIMIT_RATIO', 'CB_ATM_DRAWING_BALANCE_RATIO',
                                                              'CB_ATM_DRAWING_LIMIT_RATIO', 'CB_DRAWING_BALANCE_RATIO',
                                                              'CB_DRAWING_LIMIT_RATIO', 'CB_POS_DRAWING_BALANCE_RATIO',
                                                              'CB_POS_DRAWING_LIMIT_RATIO', 'CB_DRAWING_ATM_RATIO',
                                                              'CB_DRAWING_POS_RATIO', 'CB_MIN_INSTALLMENT_BALANCE_RATIO',
                                                              'CB_MIN_INSTALLMENT_DRAWING_RAITO', 'CB_PAYMENT_BALANCE_RATIO',
                                                              'CB_PAYMENT_LIMIT_RATIO', 'CB_RECEIVABLE_PRINCIPAL_TOTAL_RATIO',
                                                              'CB_RECEIVABLE_TOTAL_DRAWING_RATIO', 'CB_INSTAL_MIN_REG_CNT_INSTAL_MATURE_RATIO',
                                                              'CB_INSTAL_MIN_REG_LIMIT_ACTUAL_RATIO', 'CB_INSTAL_MIN_REG_PAYMENT_RATIO',
                                                              'CB_INSTAL_MIN_REG_MISSED', 'CB_DRAWING_AVG',], axis=1).drop_duplicates(keep='first')
    print('credeit_card_balance shape = {}'.format(credit_card_balance_processed.shape))
    credit_card_balance_processed.to_csv('credit_card_balance_processed.csv', index=False)