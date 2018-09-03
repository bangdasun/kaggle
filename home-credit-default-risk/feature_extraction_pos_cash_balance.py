
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
    pos_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    print(pos_cash_balance.shape)
    
with timer('Last month data'):
    pos_cash_balance_last_month = pos_cash_balance.sort_values('MONTHS_BALANCE').groupby(['SK_ID_PREV', 'SK_ID_CURR']).tail(1)
    print(pos_cash_balance_last_month.shape)
    
    pos_cash_balance_last_month = add_group_min(pos_cash_balance_last_month, cols=['SK_ID_CURR'], cname='PC_EARLIEST_MONTH', value='MONTHS_BALANCE')
    pos_cash_balance_last_month = add_group_max(pos_cash_balance_last_month, cols=['SK_ID_CURR'], cname='PC_LATEST_MONTH', value='MONTHS_BALANCE')
    pos_cash_balance_last_month = add_group_sum(pos_cash_balance_last_month, cols=['SK_ID_CURR'], cname='PC_UNIQUE_CNT_INSTALMENT_SUM', value='CNT_INSTALMENT')
    pos_cash_balance_last_month = add_group_sum(pos_cash_balance_last_month, cols=['SK_ID_CURR'], cname='PC_UNIQUE_CNT_INSTALMENT_FUTURE_SUM', value='CNT_INSTALMENT_FUTURE')
    pos_cash_balance_last_month = add_group_mean(pos_cash_balance_last_month, cols=['SK_ID_CURR'], cname='PC_UNIQUE_CNT_INSTALMENT_AVG', value='CNT_INSTALMENT')
    pos_cash_balance_last_month_contract = add_group_value_count(pos_cash_balance_last_month[['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_STATUS']],
                                                                 cols=['SK_ID_CURR', 'NAME_CONTRACT_STATUS'], value='SK_ID_PREV', prefix='PC_CURR_')
    pos_cash_balance_last_month_contract = pos_cash_balance_last_month_contract.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1)
    pos_cash_balance_last_month = pos_cash_balance_last_month.merge(pos_cash_balance_last_month_contract, on=['SK_ID_CURR'], how='left')
    pos_cash_balance_last_month['PC_PREV_ID_CNT'] = pos_cash_balance_last_month.iloc[:, -8:].sum(axis=1)
    pos_cash_balance_last_month['PC_CURR_ACTIVE_RATIO'] = pos_cash_balance_last_month['PC_CURR_ACTIVE_CNT'] / pos_cash_balance_last_month['PC_PREV_ID_CNT']
    pos_cash_balance_last_month['PC_CURR_COMPLETED_RATIO'] = pos_cash_balance_last_month['PC_CURR_COMPLETED_CNT'] / pos_cash_balance_last_month['PC_PREV_ID_CNT']
    pos_cash_balance_last_month['PC_CURR_APPROVED_RATIO'] = pos_cash_balance_last_month['PC_CURR_APPROVED_CNT'] / pos_cash_balance_last_month['PC_PREV_ID_CNT']
    pos_cash_balance_last_month['PC_CURR_RETURNED_TO_THE_STORE_RATIO'] = pos_cash_balance_last_month['PC_CURR_RETURNED_TO_THE_STORE_CNT'] / pos_cash_balance_last_month['PC_PREV_ID_CNT']
    pos_cash_balance_last_month['PC_CURR_SIGNED_RATIO'] = pos_cash_balance_last_month['PC_CURR_SIGNED_CNT'] / pos_cash_balance_last_month['PC_PREV_ID_CNT']
    pos_cash_balance_last_month = pos_cash_balance_last_month.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS', 'MONTHS_BALANCE',
                                                                    'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF'], axis=1)
    pos_cash_balance_last_month = pos_cash_balance_last_month.drop_duplicates(keep='first')
    print(pos_cash_balance_last_month.shape)

with timer('Get last 5 records dfs'):
    prev_3_id = get_prev_k_id(pos_cash_balance, k=5)
    pos_cash_balance_last_3 = pos_cash_balance.loc[pos_cash_balance['SK_ID_PREV'].isin(prev_3_id)]
    pos_cash_balance_last_3['HAS_DPD'] = (pos_cash_balance_last_3['SK_DPD'] > 0).astype(np.int32)
    pos_cash_balance_last_3['HAS_DPD_DEF'] = (pos_cash_balance_last_3['SK_DPD_DEF'] > 0).astype(np.int32)
    pos_cash_balance_last_3 = add_group_max(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'CNT_INSTALMENT_MAX', value='CNT_INSTALMENT')
    pos_cash_balance_last_3 = add_group_min(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'CNT_INSTALMENT_MIN', value='CNT_INSTALMENT')
    pos_cash_balance_last_3 = add_group_max(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'CNT_INSTALMENT_FUTURE_MAX', value='CNT_INSTALMENT_FUTURE')
    pos_cash_balance_last_3 = add_group_min(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'CNT_INSTALMENT_FUTURE_MIN', value='CNT_INSTALMENT_FUTURE')
    pos_cash_balance_last_3 = add_group_min(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_MIN', value='SK_DPD')
    pos_cash_balance_last_3 = add_group_max(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_MAX', value='SK_DPD')
    pos_cash_balance_last_3 = add_group_mean(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_AVG', value='SK_DPD')
    pos_cash_balance_last_3 = add_group_sum(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_SUM', value='SK_DPD')
    
    pos_cash_balance_last_3 = add_group_min(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_DEF_MIN', value='SK_DPD_DEF')
    pos_cash_balance_last_3 = add_group_max(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_DEF_MAX', value='SK_DPD_DEF')
    pos_cash_balance_last_3 = add_group_mean(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_DEF_AVG', value='SK_DPD_DEF')
    pos_cash_balance_last_3 = add_group_sum(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'SK_DPD_DEF_SUM', value='SK_DPD_DEF')
    
    pos_cash_balance_last_3 = add_group_sum(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'HAS_DPD'+'_SUM', value='HAS_DPD')
    pos_cash_balance_last_3 = add_group_mean(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'HAS_DPD'+'_AVG', value='HAS_DPD')
    
    pos_cash_balance_last_3 = add_group_sum(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'HAS_DPD_DEF'+'_SUM', value='HAS_DPD_DEF')
    pos_cash_balance_last_3 = add_group_mean(pos_cash_balance_last_3, cols=['SK_ID_CURR'], cname='PC_LAST_5_'+'HAS_DPD_DEF'+'_AVG', value='HAS_DPD_DEF')

with timer('Get last 10 records dfs'):
    prev_10_id = get_prev_k_id(pos_cash_balance, k=10)
    pos_cash_balance_last_10 = pos_cash_balance.loc[pos_cash_balance['SK_ID_PREV'].isin(prev_10_id)]
    pos_cash_balance_last_10['HAS_DPD'] = (pos_cash_balance_last_10['SK_DPD'] > 0).astype(np.int32)
    pos_cash_balance_last_10['HAS_DPD_DEF'] = (pos_cash_balance_last_10['SK_DPD_DEF'] > 0).astype(np.int32)
    pos_cash_balance_last_10 = add_group_max(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'CNT_INSTALMENT_MAX', value='CNT_INSTALMENT')
    pos_cash_balance_last_10 = add_group_min(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'CNT_INSTALMENT_MIN', value='CNT_INSTALMENT')
    pos_cash_balance_last_10 = add_group_max(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'CNT_INSTALMENT_FUTURE_MAX', value='CNT_INSTALMENT_FUTURE')
    pos_cash_balance_last_10 = add_group_min(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'CNT_INSTALMENT_FUTURE_MIN', value='CNT_INSTALMENT_FUTURE')
    pos_cash_balance_last_10 = add_group_min(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_MIN', value='SK_DPD')
    pos_cash_balance_last_10 = add_group_max(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_MAX', value='SK_DPD')
    pos_cash_balance_last_10 = add_group_mean(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_AVG', value='SK_DPD')
    pos_cash_balance_last_10 = add_group_sum(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_SUM', value='SK_DPD')
    pos_cash_balance_last_10 = add_group_min(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_DEF_MIN', value='SK_DPD_DEF')
    pos_cash_balance_last_10 = add_group_max(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_DEF_MAX', value='SK_DPD_DEF')
    pos_cash_balance_last_10 = add_group_mean(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_DEF_AVG', value='SK_DPD_DEF')
    pos_cash_balance_last_10 = add_group_sum(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'SK_DPD_DEF_SUM', value='SK_DPD_DEF')
    pos_cash_balance_last_10 = add_group_sum(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'HAS_DPD'+'_SUM', value='HAS_DPD')
    pos_cash_balance_last_10 = add_group_mean(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'HAS_DPD'+'_AVG', value='HAS_DPD')
    pos_cash_balance_last_10 = add_group_sum(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'HAS_DPD_DEF'+'_SUM', value='HAS_DPD_DEF')
    pos_cash_balance_last_10 = add_group_mean(pos_cash_balance_last_10, cols=['SK_ID_CURR'], cname='PC_LAST_10_'+'HAS_DPD_DEF'+'_AVG', value='HAS_DPD_DEF')
    
with timer('Post processing pos_cash_balance_last_k'):
    pos_cash_balance_last_3 = pos_cash_balance_last_3.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
                                                            'CNT_INSTALMENT_FUTURE', 'NAME_CONTRACT_STATUS',
                                                            'SK_DPD', 'SK_DPD_DEF', 'HAS_DPD', 'HAS_DPD_DEF'], axis=1).drop_duplicates(keep='first')
    
    pos_cash_balance_last_10 = pos_cash_balance_last_10.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
                                                              'CNT_INSTALMENT_FUTURE', 'NAME_CONTRACT_STATUS',
                                                              'SK_DPD', 'SK_DPD_DEF', 'HAS_DPD', 'HAS_DPD_DEF'], axis=1).drop_duplicates(keep='first')
    
with timer('Find last record'):
    pos_cash_balance = add_group_max(pos_cash_balance, cols=['SK_ID_CURR'], cname='LAST_MONTH_BALANCE', value='MONTHS_BALANCE')
    pos_cash_balance['IS_LAST_MONTHS'] = pos_cash_balance['MONTHS_BALANCE'] == pos_cash_balance['LAST_MONTH_BALANCE']
    
with timer('Simple features'):
    pos_cash_balance['HAS_DPD'] = (pos_cash_balance['SK_DPD'] > 0).astype(np.int32)
    pos_cash_balance['HAS_DPD_DEF'] = (pos_cash_balance['SK_DPD_DEF'] > 0).astype(np.int32)
    
with timer('Extract last record features'):
    pos_cash_balance_last = pos_cash_balance.loc[pos_cash_balance['IS_LAST_MONTHS']]
    pos_cash_balance_last.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'IS_LAST_MONTHS', 'LAST_MONTH_BALANCE'], axis=1, inplace=True)
    pos_cash_balance_last.columns = ['SK_ID_CURR'] + ['PC_LAST_' + c for c in pos_cash_balance_last.columns.tolist()[1:]]
    pos_cash_balance_last['PC_LAST_NAME_CONTRACT_STATUS'] = LabelEncoder().fit_transform(pos_cash_balance_last['PC_LAST_NAME_CONTRACT_STATUS'].values)
    print('pos_cash_balance_last shape = {}'.format(pos_cash_balance_last.shape))
    pos_cash_balance_last = pos_cash_balance_last.groupby(['SK_ID_CURR']).mean().reset_index()
    print('pos_cash_balance_last shape = {}'.format(pos_cash_balance_last.shape))
    
with timer("Extracted groupby features"):
    pos_cash_balance = add_group_max(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'CNT_INSTALMENT_MAX', value='CNT_INSTALMENT') #-
    pos_cash_balance = add_group_min(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'CNT_INSTALMENT_MIN', value='CNT_INSTALMENT') #-
    pos_cash_balance = add_group_max(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'CNT_INSTALMENT_FUTURE_MAX', value='CNT_INSTALMENT_FUTURE') #-
    pos_cash_balance = add_group_min(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'CNT_INSTALMENT_FUTURE_MIN', value='CNT_INSTALMENT_FUTURE') #-
    pos_cash_balance = add_group_min(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_MIN', value='SK_DPD')
    pos_cash_balance = add_group_max(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_MAX', value='SK_DPD')
    pos_cash_balance = add_group_mean(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_AVG', value='SK_DPD')
    pos_cash_balance = add_group_sum(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_SUM', value='SK_DPD')    
    pos_cash_balance = add_group_min(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_DEF_MIN', value='SK_DPD_DEF')
    pos_cash_balance = add_group_max(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_DEF_MAX', value='SK_DPD_DEF')
    pos_cash_balance = add_group_mean(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_DEF_AVG', value='SK_DPD_DEF')
    pos_cash_balance = add_group_sum(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'SK_DPD_DEF_SUM', value='SK_DPD_DEF')
    pos_cash_balance = add_group_sum(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'HAS_DPD'+'_SUM', value='HAS_DPD')
    pos_cash_balance = add_group_mean(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'HAS_DPD'+'_AVG', value='HAS_DPD')
    pos_cash_balance = add_group_sum(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'HAS_DPD_DEF'+'_SUM', value='HAS_DPD_DEF')
    pos_cash_balance = add_group_mean(pos_cash_balance, cols=['SK_ID_CURR'], cname='PC_'+'HAS_DPD_DEF'+'_AVG', value='HAS_DPD_DEF')
    
with timer('Extract NAME_CONTRACT_STATUS'):
    pos_cash_balance_contract_status = add_group_value_count(pos_cash_balance[['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_STATUS']],
                                                             cols=['SK_ID_CURR', 'NAME_CONTRACT_STATUS'],
                                                             value='SK_ID_PREV', prefix='PC_')
    pos_cash_balance_contract_status.columns = pos_cash_balance_contract_status.columns.tolist()[:3] + [c for c in pos_cash_balance_contract_status.columns[3:]]
    pos_cash_balance = pd.concat([pos_cash_balance, pos_cash_balance_contract_status.iloc[:, 3:]], axis=1)
    
with timer('Generate pos_cash_balance_processed'):
    pos_cash_balance_processed = pos_cash_balance.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 
                                                        'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF', 'HAS_DPD', 'HAS_DPD_DEF'], axis=1)
    pos_cash_balance_processed = pos_cash_balance_processed.drop_duplicates(keep='first')
    pos_cash_balance_processed = pos_cash_balance_processed.merge(pos_cash_balance_last, on=['SK_ID_CURR'], how='left')
    pos_cash_balance_processed.drop(['LAST_MONTH_BALANCE', 'IS_LAST_MONTHS'], axis=1, inplace=True)
    pos_cash_balance_processed = pos_cash_balance_processed.drop_duplicates(keep='first')
    print('pos_cash_balance shape = {}'.format(pos_cash_balance_processed.shape))
    pos_cash_balance_processed = pos_cash_balance_processed.merge(pos_cash_balance_last_month, on=['SK_ID_CURR'], how='left')
    print('pos_cash_balance shape = {}'.format(pos_cash_balance_processed.shape))
    pos_cash_balance_processed = pos_cash_balance_processed.merge(pos_cash_balance_last_3, on=['SK_ID_CURR'], how='left')
    print('pos_cash_balance shape = {}'.format(pos_cash_balance_processed.shape))
    pos_cash_balance_processed = pos_cash_balance_processed.merge(pos_cash_balance_last_10, on=['SK_ID_CURR'], how='left')
    print('pos_cash_balance shape = {}'.format(pos_cash_balance_processed.shape))
    pos_cash_balance_processed.to_csv('pos_cash_balance_processed.csv', index=False)