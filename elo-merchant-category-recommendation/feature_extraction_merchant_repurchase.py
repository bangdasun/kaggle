
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer

with timer('Load data'):
    hist = pd.read_csv('hist_transac_processed.csv', usecols=['card_id', 'merchant_id'])
    print('historical transaction data: {}'.format(hist.shape))

with timer('Merchant customer count'):
    hist_merchant = hist.groupby(['merchant_id']).size().reset_index()
    hist_merchant.columns = ['merchant_id', 'merchant_customer_count']
    print(hist_merchant.shape)

with timer('Merchant repurchase customer count'):
    hist_merchant_card = hist.groupby(['merchant_id', 'card_id']).size().reset_index()
    hist_merchant_card.columns = ['merchant_id', 'card_id', 'customer_visit_count']
    print(hist_merchant_card.shape)
    hist_merchant_card = hist_merchant_card.loc[hist_merchant_card['customer_visit_count'] > 1]
    print(hist_merchant_card.shape)

    # binary count
    hist_merchant_repurchase_binary = hist_merchant_card.groupby(['merchant_id']).size().reset_index()
    hist_merchant_repurchase_binary.columns = ['merchant_id', 'revisited_customers']
    hist_merchant_repurchase_binary['revisited_customers'].fillna(0.0, inplace=True)
    print(hist_merchant_repurchase_binary.shape)
    print(hist_merchant_repurchase_binary.head())

    # exact count
    hist_merchant_repurchase_exact = hist_merchant_card.groupby(['merchant_id'])['customer_visit_count'].sum().reset_index()
    hist_merchant_repurchase_exact.columns = ['merchant_id', 'revisited_count']
    hist_merchant_repurchase_exact['revisited_count'].fillna(0.0, inplace=True)
    print(hist_merchant_repurchase_exact.shape)
    print(hist_merchant_repurchase_exact.head())

with timer('Concatenate'):
    hist_merchant = hist_merchant.merge(hist_merchant_repurchase_binary, on=['merchant_id'], how='left')
    hist_merchant = hist_merchant.merge(hist_merchant_repurchase_exact, on=['merchant_id'], how='left')
    hist_merchant['repurchase_customer_ratio'] = hist_merchant['revisited_customers'].values / hist_merchant['merchant_customer_count'].values
    hist_merchant['repurchase_ratio'] = hist_merchant['revisited_count'].values / hist_merchant['merchant_customer_count'].values
    print(hist_merchant.shape)
    print(hist_merchant.head())

with timer('Combine with transactions'):
    hist = hist.merge(hist_merchant, on=['merchant_id'], how='left')
    hist_feats = pd.DataFrame(hist.groupby(['card_id']).size()).reset_index()
    hist_feats.columns = ['card_id', 'hist_transac_count']
    for m in ['mean', 'std', 'max', 'min']:
        hist_feats['merchant_repurchase_customer_ratio_{}'.format(m)] = hist.groupby(['card_id'])['repurchase_customer_ratio'].agg([m]).values
        hist_feats['merchant_repurchase_ratio_{}'.format(m)] = hist.groupby(['card_id'])['repurchase_ratio'].agg([m]).values

with timer('Save merchant repurchase features'):
    hist_feats.drop(['hist_transac_count'], axis=1, inplace=True)
    print('hist_feats: {}'.format(hist_feats.shape))
    print(hist_feats.head())
    hist_feats.to_csv('merchant_repurchase_rates.csv', index=False)