
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
    installments_payments = pd.read_csv('installments_payments.csv')
    print(installments_payments.shape)
    
with timer('Find last instalment'):
    installments_payments = add_group_max(installments_payments, cols=['SK_ID_CURR'], cname='LAST_INSTALMENT', value='DAYS_INSTALMENT')
    installments_payments['IS_LAST_INSTALMENT'] = (installments_payments['DAYS_INSTALMENT'] == installments_payments['LAST_INSTALMENT'])
    installments_payments.loc[installments_payments['IS_LAST_INSTALMENT']][['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(keep='last').to_csv('last_instalment_id.csv', index=False)
    
with timer('Extract last instalment features'):
    installments_payments_last = installments_payments.loc[installments_payments['IS_LAST_INSTALMENT']]
    installments_payments_last.drop(['SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'LAST_INSTALMENT', 'IS_LAST_INSTALMENT'], axis=1, inplace=True)
    installments_payments_last.columns = ['SK_ID_CURR'] + ['IP_LAST_'+c for c in installments_payments_last.columns.tolist()[1:]]
    print('installments_payments_last shape = {}'.format(installments_payments_last.shape))
    installments_payments_last = installments_payments_last.groupby(['SK_ID_CURR']).mean().reset_index()
    print('installments_payments_last shape = {}'.format(installments_payments_last.shape))

with timer('Simple features'):
    installments_payments_last['IP_LAST_DPD'] = installments_payments_last['IP_LAST_DAYS_ENTRY_PAYMENT'] - installments_payments_last['IP_LAST_DAYS_INSTALMENT']
    installments_payments_last['IP_IS_LAST_LATE_DPD'] = installments_payments_last['IP_LAST_DPD'].apply(lambda x: 1 if x > 0 else 0)
    installments_payments_last['IP_IS_LAST_EARLY_DPD'] = installments_payments_last['IP_LAST_DPD'].apply(lambda x: 1 if x < 0 else 0)
    installments_payments_last['IP_IS_LAST_OT_DPD'] = installments_payments_last['IP_LAST_DPD'].apply(lambda x: 1 if x == 0 else 0)
    installments_payments_last['IP_LAST_DPD_AMT'] = installments_payments_last['IP_LAST_AMT_PAYMENT'] - installments_payments_last['IP_LAST_AMT_INSTALMENT']
    installments_payments_last['IP_LAST_IS_INSUFFICIENT_AMT'] = installments_payments_last['IP_LAST_DPD_AMT'].apply(lambda x: 1 if x < 0 else 0)
    installments_payments_last['IP_LAST_INSUFFICIENT_AMT'] = installments_payments_last['IP_LAST_DPD_AMT'].apply(lambda x: x if x < 0 else 0)
    
with timer('Groupby features'):
    installments_payments = add_group_nunique(installments_payments, cols=['SK_ID_CURR'], cname='IP_'+'INSTALMENT_VERSION_NUNIQUE', value='NUM_INSTALMENT_VERSION')
    installments_payments = add_group_nunique(installments_payments, cols=['SK_ID_CURR'], cname='IP_'+'INSTALMENT_NUMBER_NUNIQUE', value='NUM_INSTALMENT_NUMBER')
    installments_payments = add_group_count(installments_payments, cols=['SK_ID_CURR'], cname='IP_'+'INSTALLMENT_COUNT', value='NUM_INSTALMENT_VERSION')
    installments_payments['IP_'+'INSTALMENT_UNIQUE_VERSION_RATIO'] = installments_payments['IP_'+'INSTALMENT_VERSION_NUNIQUE'] / installments_payments['IP_'+'INSTALLMENT_COUNT']
    installments_payments['IP_'+'INSTALMENT_UNIQUE_NUMBER_RATIO'] = installments_payments['IP_'+'INSTALMENT_NUMBER_NUNIQUE'] / installments_payments['IP_'+'INSTALLMENT_COUNT']
    installments_payments['IP_'+'INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT'] 
    installments_payments['IP_'+'INSTALMENT_PAYMENT_DATE_RATIO'] = installments_payments['DAYS_INSTALMENT'] / installments_payments['DAYS_ENTRY_PAYMENT'] 
    installments_payments['IP_'+'INSTALMENT_PAYMENT_RATIO'] = installments_payments['AMT_PAYMENT'] / installments_payments['AMT_INSTALMENT'] 
    installments_payments['IP_'+'INSTALMENT_PAYMENT_DIFF'] = installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']
    installments_payments['IP_DPD'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
    installments_payments['IP_IS_LATE_DPD'] = installments_payments['IP_DPD'].apply(lambda x: 1 if x > 0 else 0)
    installments_payments['IP_IS_EARLY_DPD'] = installments_payments['IP_DPD'].apply(lambda x: 1 if x < 0 else 0)
    installments_payments['IP_IS_OT_DPD'] = installments_payments['IP_DPD'].apply(lambda x: 1 if x == 0 else 0)
    installments_payments['IP_LATE_DPD'] = installments_payments['IP_DPD'].apply(lambda x: x if x > 0 else 0)
    installments_payments['IP_EARLY_DPD'] = installments_payments['IP_DPD'].apply(lambda x: x if x < 0 else 0)
    installments_payments['IP_DPD_AMT'] = installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']
    installments_payments['IP_DPD_INSUFFICIENT_AMT'] = installments_payments['IP_DPD_AMT'].apply(lambda x: x if x < 0 else 0)
    installments_payments['IP_DPD_IS_INSUFFICIENT_AMT'] = installments_payments['IP_DPD_AMT'].apply(lambda x: 1 if x < 0 else 0)
    
    agg_func_map = {
        "mean": add_group_mean,
        "max" : add_group_max,
        "min" : add_group_min,
        "sum" : add_group_sum
        "std" : add_group_std,        
    }
    
    agg_config = [
        (["SK_ID_CURR"], [("IP_LATE_DPD", "sum"),
                          ("IP_LATE_DPD", "mean"),
                          ("IP_LATE_DPD", "max"),
                          ("IP_IS_LATE_DPD", "sum"),
                          ("IP_IS_LATE_DPD", "mean"),
                          ("IP_IS_EARLY_DPD", "sum"),
                          ("IP_IS_EARLY_DPD", "mean"),
                          ("IP_IS_OT_DPD", "sum"),
                          ("IP_IS_OT_DPD", "sum"),
                          ("IP_DPD_INSUFFICIENT_AMT", "sum"),
                          ("IP_DPD_INSUFFICIENT_AMT", "min"),
                          ("IP_DPD_IS_INSUFFICIENT_AMT", "sum"),
                          ("IP_DPD_IS_INSUFFICIENT_AMT", "min"),
                          ("DAYS_INSTALMENT", "mean"),
                          ("DAYS_INSTALMENT", "max"),
                          ("DAYS_INSTALMENT", "min"),
                          ("AMT_INSTALMENT", "mean"),
                          ("AMT_INSTALMENT", "max"),
                          ("AMT_INSTALMENT", "min"),
                          ("AMT_INSTALMENT", "sum"),
                          ("IP_INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF", "mean"),
                          ("IP_INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF", "max"),
                          ("IP_INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF", "min"),
                          ("IP_INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF", "sum"),
                          ("IP_INSTALMENT_PAYMENT_DATE_RATIO", "mean"),
                          ("IP_INSTALMENT_PAYMENT_DATE_RATIO", "min"),
                          ("IP_INSTALMENT_PAYMENT_DATE_RATIO", "std"),
                          ("IP_INSTALMENT_PAYMENT_DIFF", "mean"),
                          ("IP_INSTALMENT_PAYMENT_DIFF", "max"),
                          ("IP_INSTALMENT_PAYMENT_DIFF", "sum"),
                          ("IP_INSTALMENT_PAYMENT_DIFF", "std"),
                          ("IP_INSTALMENT_PAYMENT_RATIO", "mean"),
                          ("IP_INSTALMENT_PAYMENT_RATIO", "max"),
                          ("IP_INSTALMENT_PAYMENT_RATIO", "std")])
    ]
    
    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            installments_payments = agg_func_map[agg_feat[1]](installments_payments, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
            
with timer('Generate installments_payments_processed'):
    installments_payments = add_group_mean(installments_payments, cols=['SK_ID_CURR'], cname='IP_DPD_INSUFFICIENT_AMT' + '_AVG', value='IP_DPD_INSUFFICIENT_AMT')
    installments_payments_processed = installments_payments.drop(['SK_ID_PREV', 'NUM_INSTALMENT_VERSION',
                                                                  'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT',
                                                                  'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT', 
                                                                  'IP_INSTALMENT_INSTALMENT_ENTRY_DATE_DIFF', 'IP_INSTALMENT_PAYMENT_DATE_RATIO',
                                                                  'IP_INSTALMENT_PAYMENT_RATIO', 'IP_INSTALMENT_PAYMENT_DIFF', 'IP_DPD', 'IP_DPD_AMT',
                                                                  'IP_IS_LATE_DPD', 'IP_IS_EARLY_DPD', 'IP_IS_OT_DPD',
                                                                  'IP_LATE_DPD', 'IP_EARLY_DPD', 'IP_DPD_INSUFFICIENT_AMT', 'IP_DPD_IS_INSUFFICIENT_AMT',
                                                                  'LAST_INSTALMENT', 'IS_LAST_INSTALMENT'], axis=1)
    installments_payments_processed = installments_payments_processed.drop_duplicates(keep='first')
    print('installments_payments_processed shape = {}'.format(installments_payments_processed.shape))
    installments_payments_processed = installments_payments_processed.merge(installments_payments_last, on=['SK_ID_CURR'], how='left')
    print('installments_payments_processed shape = {}'.format(installments_payments_processed.shape))
    
with timer('Extra features on installments_payments_processed'):
    installments_payments_processed['IP_LAST_AMT_INSTALMENT_AVG_RATIO'] = installments_payments_processed['IP_LAST_AMT_INSTALMENT'] / installments_payments_processed['IP_AMT_INSTALMENT_AVG']
    installments_payments_processed['IP_LAST_AMT_INSTALMENT_MAX_RATIO'] = installments_payments_processed['IP_LAST_AMT_INSTALMENT'] / installments_payments_processed['IP_AMT_INSTALMENT_MAX']
    installments_payments_processed['IP_LAST_AMT_INSTALMENT_MIN_RATIO'] = installments_payments_processed['IP_LAST_AMT_INSTALMENT'] / installments_payments_processed['IP_AMT_INSTALMENT_MIN']
    installments_payments_processed['IP_LAST_DPD_AVG_DIFF'] = installments_payments_processed['IP_LAST_DPD'] - installments_payments_processed['IP_LATE_DPD_AVG']
    installments_payments_processed['IP_LAST_DPD_AMT_AVG_DIFF'] = installments_payments_processed['IP_LAST_DPD_AMT'] - installments_payments_processed['IP_DPD_INSUFFICIENT_AMT_AVG']
    print('installments_payments_processed shape = {}'.format(installments_payments_processed.shape))
    installments_payments_processed.to_csv('installments_payments_processed.csv', index=False)