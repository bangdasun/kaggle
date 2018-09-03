
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
    previous_application = pd.read_csv('previous_application.csv')
    print(previous_application.shape)
    
with timer('Preprocessing'):
    previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    
with timer('Simple features'):
    previous_application['PA_CREDIT_RATIO'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']
    previous_application['PA_ANNUITY_CREDIT_RATIO'] = previous_application['AMT_ANNUITY'] / previous_application['AMT_CREDIT']
    previous_application['PA_APP_ANNUITY_RATIO'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_ANNUITY']
    previous_application['PA_APP_PRICE_RATIO'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_GOODS_PRICE']
    previous_application['PA_DOWN_PAY_APP_RATIO'] = previous_application['AMT_DOWN_PAYMENT'] / previous_application['AMT_APPLICATION']
    
    gp_feat = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
               'PA_CREDIT_RATIO', 'PA_ANNUITY_CREDIT_RATIO', 'PA_APP_ANNUITY_RATIO', 'PA_APP_PRICE_RATIO', 'PA_DOWN_PAY_APP_RATIO',
               'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'DAYS_DECISION', 'CNT_PAYMENT']
    
with timer('Split to approved / refused parts'):
    previous_application_approved = previous_application.loc[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
    previous_application_refused = previous_application.loc[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
    print(previous_application_approved.shape, previous_application_refused.shape)
    
with timer('Load last instalment id'):
    last_app_id = pd.read_csv('last_instalment_id.csv')
    previous_application_approved = previous_application_approved.merge(last_app_id, on=['SK_ID_CURR'], how='left')
    previous_application_last_approved = previous_application_approved.loc[previous_application_approved['SK_ID_PREV_x'] == previous_application_approved['SK_ID_PREV_y']]
    print(previous_application_last_approved.shape)
    
with timer('Process last approved application'):
    previous_application_last_approved = previous_application_last_approved.drop(['SK_ID_PREV_x', 'SK_ID_PREV_y', 
                                                                                  'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
                                                                                  'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
                                                                                  'NFLAG_LAST_APPL_IN_DAY', 'NAME_CONTRACT_STATUS',
                                                                                  'CODE_REJECT_REASON'], axis=1)
    
    cate_feat = ['NAME_CASH_LOAN_PURPOSE', 'NAME_PAYMENT_TYPE', 'NAME_TYPE_SUITE',
                 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
                 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA',
                 'NAME_SELLER_INDUSTRY', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY',
                 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']
    
    for feat in cate_feat:
        previous_application_last_approved[feat] = LabelEncoder().fit_transform(previous_application_last_approved[feat].astype(str).values)
    print(previous_application_last_approved.shape)
    previous_application_last_approved.columns = ['SK_ID_CURR'] + ['PA_LAST_APPR_'+c for c in previous_application_last_approved.columns.tolist()[1:]]
    previous_application_last_approved = previous_application_last_approved.groupby('SK_ID_CURR').first().reset_index()
    
    print(previous_application_last_approved.shape)
    
with timer('Extract groupby features'):
    for i in gp_feat:
        previous_application = add_group_mean(previous_application, cols=['SK_ID_CURR'], cname='PA_{}_AVG'.format(i), value=i)
        previous_application = add_group_max(previous_application, cols=['SK_ID_CURR'], cname='PA_{}_MAX'.format(i), value=i)
        previous_application = add_group_min(previous_application, cols=['SK_ID_CURR'], cname='PA_{}_MIN'.format(i), value=i)
        
with timer('Extract value_counts features'):
    previous_application = add_group_value_count(previous_application, cols=['SK_ID_CURR', 'NAME_CONTRACT_STATUS'], value='SK_ID_PREV', prefix='PA_')
    
with timer('Simple features'):
    previous_application['PA_APP_CNT'] = previous_application['PA_APPROVED_CNT'] + \
                                           previous_application['PA_CANCELED_CNT'] + \
                                           previous_application['PA_REFUSED_CNT'] + \
                                           previous_application['PA_UNUSED_OFFER_CNT']

    previous_application['PA_APPROVED_RATIO'] = previous_application['PA_APPROVED_CNT'] / previous_application['PA_APP_CNT']
    previous_application['PA_REFUSED_RATIO'] = previous_application['PA_REFUSED_CNT'] / previous_application['PA_APP_CNT']
    
with timer('Generate previous_application_processed'):
    previous_application_processed = previous_application.drop(['SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY', 
                                                                'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
                                                                'AMT_GOODS_PRICE', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                                                                'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT',
                                                                'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'NAME_CASH_LOAN_PURPOSE',
                                                                'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
                                                                'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
                                                                'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
                                                                'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY',
                                                                'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
                                                                'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                                                                'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL',
                                                                'PA_CREDIT_RATIO', 'PA_ANNUITY_CREDIT_RATIO', 'PA_APP_ANNUITY_RATIO',
                                                                'PA_APP_PRICE_RATIO', 'PA_DOWN_PAY_APP_RATIO'], axis=1).drop_duplicates(keep='first')
    
    previous_application_processed = previous_application_processed.merge(previous_application_last_approved, on=['SK_ID_CURR'], how='left')
    previous_application_processed.to_csv('previous_application_processed.csv', index=False)