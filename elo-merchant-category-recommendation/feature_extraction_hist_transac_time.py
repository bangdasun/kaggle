
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer

with timer('Load data'):
    hist = pd.read_csv('hist_transac_processed.csv')
    print('historical transaction data: {}'.format(hist.shape))

with timer('Get feature dataframe base'):
    hist_feats = pd.DataFrame(hist.groupby(['card_id']).size()).reset_index()
    hist_feats.columns = ['card_id', 'hist_transac_count']

with timer('Transform purchase amount'):
    hist['purchase_amount'] = np.round(hist['purchase_amount'] / 0.00150265118 + 497.06, 2)

with timer('Transaction time features'):
    for m in ['nunique', 'mean', 'std', 'min', 'skew']:
        hist_feats['hist_transac_monthlag_{}'.format(m)] = hist.groupby(['card_id'])['month_lag'].agg([m]).values

    hist_feats['hist_purchase_date_last'] = hist.groupby(['card_id'])['purchase_date'].max().values
    hist_feats['hist_purchase_date_first'] = hist.groupby(['card_id'])['purchase_date'].min().values
    hist_feats['hist_purchase_date_diff_day'] = (pd.to_datetime(hist_feats['hist_purchase_date_last']) - pd.to_datetime(hist_feats['hist_purchase_date_first'])).dt.days.values
    hist_feats['hist_purchase_count_ratio'] = hist_feats['hist_transac_count'].values / (1. + hist_feats['hist_purchase_date_diff_day'].values)

    hist['purchase_date'] = pd.to_datetime(hist['purchase_date'])
    hist['is_weekend'] = (hist['purchase_date'].dt.weekday >= 5).astype(int)
    hist_feats['hist_transac_purchase_weekend_count'] = hist.groupby(['card_id'])['is_weekend'].sum().values
    hist_feats['hist_transac_purchase_weekend_mean'] = hist.groupby(['card_id'])['is_weekend'].mean().values

    hist_feats = hist_feats.merge(hist[['card_id', 'reference_month']].drop_duplicates(), on='card_id', how='left')
    hist_feats['reference_month'] = pd.to_datetime(hist_feats['reference_month'])

    hist['month_diff'] = (pd.to_datetime('2018-12-31') - pd.to_datetime(hist['purchase_date'])).dt.days // 30
    hist['month_diff'] += hist['month_lag']
    for m in ['mean', 'std', 'min', 'max']:
        hist_feats['hist_month_diff_{}'.format(m)] = hist.groupby(['card_id'])['month_diff'].agg([m]).values

    hist['weekofyear'] = hist['purchase_date'].dt.weekofyear.values
    hist['dayofweek'] = hist['purchase_date'].dt.dayofweek.values
    hist['hour'] = hist['purchase_date'].dt.hour.values
    for m in ['nunique', 'mean', 'min', 'max']:
        for c in ['weekofyear', 'dayofweek', 'hour']:
            hist_feats['hist_transac_{}_{}'.format(c, m)] = hist.groupby(['card_id'])[c].agg([m]).values

    hist['duration'] = hist['purchase_amount'].values * hist['month_diff'].values
    hist['amount_month_ratio'] = hist['purchase_amount'].values / (1. + hist['month_diff'].values)
    for m in ['mean', 'std', 'min', 'max', 'skew']:
        hist_feats['hist_transac_duration_{}'.format(m)] = hist.groupby(['card_id'])['duration'].agg([m]).values
        hist_feats['hist_transac_amount_month_ratio_{}'.format(m)] = hist.groupby(['card_id'])['amount_month_ratio'].agg([m]).values

with timer('Recent time transaction features'):
    for c in ['month_lag=0', 'month_lag=-1', 'month_lag=-2']:
        hist_feats['hist_transac_{}_count'.format(c)] = hist.groupby(['card_id'])[c].sum().values
        hist_feats['hist_transac_{}_mean'.format(c)] = hist.groupby(['card_id'])[c].mean().values

    hist_feats['hist_transac_monthlag_0_-1_ratio'] = hist_feats['hist_transac_month_lag=0_count'].values / (1. + hist_feats['hist_transac_month_lag=-1_count'].values)
    hist_feats['hist_transac_monthlag_0_-2_ratio'] = hist_feats['hist_transac_month_lag=0_count'].values / (1. + hist_feats['hist_transac_month_lag=-2_count'].values)

    hist_feats['hist_transac_last_3_mon_count'] = hist_feats['hist_transac_month_lag=0_count'].values \
                                                + hist_feats['hist_transac_month_lag=-1_count'].values \
                                                + hist_feats['hist_transac_month_lag=-2_count'].values
    hist_feats['hist_transac_last_3_mon_ratio'] = hist_feats['hist_transac_last_3_mon_count'].values / (1. + hist_feats['hist_transac_count'].values)

with timer('Time decay features'):
    hist = hist.sort_values('purchase_date')
    tmp_df = hist.groupby(['card_id']).size().reset_index()
    tmp_df.columns = ['card_id', 'hist_transac_count']
    hist = hist.merge(tmp_df, on=['card_id'], how='left')
    hist['transac_seq_num'] = hist.groupby(['card_id']).cumcount() + 1
    hist['transac_seq_num_desc'] = hist['hist_transac_count'] - hist['transac_seq_num'] - 1
    hist['transac_decay'] = 0.8 ** hist['transac_seq_num_desc'].values
    for m in ['mean', 'sum']:
        hist_feats['hist_transac_decay_{}'.format(m)] = hist.groupby(['card_id'])['transac_decay'].agg([m]).values

with timer('Purchase time delta features'):
    hist['prev_1_purchase_date'] = hist.groupby(['card_id'])['purchase_date'].shift(1)
    hist['purchase_date_diff_days'] = (hist['purchase_date'] - hist['prev_1_purchase_date']).dt.days.values
    hist['purchase_date_diff_seconds'] = hist['purchase_date_diff_days'].values * 24 * 3600
    hist['purchase_date_diff_seconds'] += (hist['purchase_date'] - hist['prev_1_purchase_date']).dt.seconds.values
    hist['purchase_date_diff_hours'] = hist['purchase_date_diff_seconds'].values // 3600

    for m in ['mean', 'std', 'max', 'min']:
        hist_feats['hist_transac_purchase_date_diff_sec_{}'.format(m)] = hist.groupby(['card_id'])['purchase_date_diff_seconds'].agg([m]).values
        hist_feats['hist_transac_purchase_date_diff_day_{}'.format(m)] = hist.groupby(['card_id'])['purchase_date_diff_days'].agg([m]).values
        hist_feats['hist_transac_purchase_date_diff_hour_{}'.format(m)] = hist.groupby(['card_id'])['purchase_date_diff_hours'].agg([m]).values

with timer('Special date influence'):
    hist['ChristmasDay_2017'] = (pd.to_datetime('2017-12-25') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    hist['FathersDay_2017'] = (pd.to_datetime('2017-08-13') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    hist['ChildrenDay_2017'] = (pd.to_datetime('2017-10-12') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    hist['BlackFriday_2017'] = (pd.to_datetime('2017-11-24') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    hist['ValentineDay_2017'] = (pd.to_datetime('2017-06-12') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    hist['MothersDay_2018'] = (pd.to_datetime('2018-05-13') - hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    for c in ['ChristmasDay_2017', 'FathersDay_2017', 'ChildrenDay_2017',
              'BlackFriday_2017', 'ValentineDay_2017', 'MothersDay_2018']:
        hist_feats['hist_transac_{}_mean'.format(c)] = hist.groupby(['card_id'])[c].mean().values

with timer('Save historical transaction time features'):
    hist_feats.drop(['hist_transac_count'], axis=1, inplace=True)
    print('hist_feats: {}'.format(hist_feats.shape))
    hist_feats.to_csv('hist_transac_time.csv', index=False)