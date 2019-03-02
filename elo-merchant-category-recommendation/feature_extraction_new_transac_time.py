
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer

with timer('Load data'):
    new = pd.read_csv('new_transac_processed.csv')
    print('new transaction data: {}'.format(new.shape))

with timer('Get feature dataframe base'):
    new_feats = pd.DataFrame(new.groupby(['card_id']).size()).reset_index()
    new_feats.columns = ['card_id', 'new_transac_count']

with timer('Transform purchase amount'):
    new['purchase_amount'] = np.round(new['purchase_amount'] / 0.00150265118 + 497.06, 2)

with timer('Transaction time features'):
    for m in ['mean', 'std', 'max']:
        new_feats['new_transac_monthlag_{}'.format(m)] = new.groupby(['card_id'])['month_lag'].agg([m]).values

    new_feats['new_purchase_date_last'] = new.groupby(['card_id'])['purchase_date'].max().values
    new_feats['new_purchase_date_first'] = new.groupby(['card_id'])['purchase_date'].min().values
    new_feats['new_purchase_date_diff_day'] = (pd.to_datetime(new_feats['new_purchase_date_last']) - pd.to_datetime(new_feats['new_purchase_date_first'])).dt.days.values
    new_feats['new_purchase_count_ratio'] = new_feats['new_transac_count'].values / (1. + new_feats['new_purchase_date_diff_day'].values)

    new['purchase_date'] = pd.to_datetime(new['purchase_date'])
    new['is_weekend'] = (new['purchase_date'].dt.weekday >= 5).astype(int)
    new_feats['new_transac_purchase_weekend_count'] = new.groupby(['card_id'])['is_weekend'].sum().values
    new_feats['new_transac_purchase_weekend_mean'] = new.groupby(['card_id'])['is_weekend'].mean().values

    new['month_diff'] = (pd.to_datetime('2018-12-31') - pd.to_datetime(new['purchase_date'])).dt.days // 30
    new['month_diff'] += new['month_lag']
    new_feats['new_month_diff_mean'] = new.groupby(['card_id'])['month_diff'].mean().values

    new['duration'] = new['purchase_amount'].values * new['month_diff'].values
    new['amount_month_ratio'] = new['purchase_amount'].values / (1. + new['month_diff'].values)
    for m in ['mean', 'std', 'min', 'max', 'skew']:
        new_feats['new_transac_duration_{}'.format(m)] = new.groupby(['card_id'])['duration'].agg([m]).values
        new_feats['new_transac_amount_month_ratio_{}'.format(m)] = new.groupby(['card_id'])['amount_month_ratio'].agg([m]).values

with timer('Recent time transaction features'):
    for c in ['month_lag=1', 'month_lag=2']:
        new_feats['new_transac_{}_count'.format(c)] = new.groupby(['card_id'])[c].sum().values
        new_feats['new_transac_{}_mean'.format(c)] = new.groupby(['card_id'])[c].mean().values

    new_feats['new_transac_month_lag=1_2_ratio'] = new_feats['new_transac_month_lag=1_count'].values / (1. + new_feats['new_transac_month_lag=2_count'].values)

with timer('Purchase time delta features'):
    new = new.sort_values('purchase_date')
    new['prev_1_purchase_date'] = new.groupby(['card_id'])['purchase_date'].shift(1)
    new['purchase_date_diff_days'] = (new['purchase_date'] - new['prev_1_purchase_date']).dt.days.values
    new['purchase_date_diff_seconds'] = new['purchase_date_diff_days'].values * 24 * 3600
    new['purchase_date_diff_seconds'] += (new['purchase_date'] - new['prev_1_purchase_date']).dt.seconds.values
    new['purchase_date_diff_hours'] = new['purchase_date_diff_seconds'].values // 3600

    for m in ['mean', 'std', 'max', 'min']:
        new_feats['new_transac_purchase_date_diff_sec_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_seconds'].agg([m]).values
        new_feats['new_transac_purchase_date_diff_day_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_days'].agg([m]).values
        new_feats['new_transac_purchase_date_diff_hour_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_hours'].agg([m]).values

    new['prev_2_purchase_date'] = new.groupby(['card_id'])['purchase_date'].shift(2)
    new['purchase_date_diff_2_days'] = (new['purchase_date'] - new['prev_2_purchase_date']).dt.days.values
    new['purchase_date_diff_2_seconds'] = new['purchase_date_diff_2_days'].values * 24 * 3600
    new['purchase_date_diff_2_seconds'] += (new['purchase_date'] - new['prev_2_purchase_date']).dt.seconds.values
    new['purchase_date_diff_2_hours'] = new['purchase_date_diff_2_seconds'].values // 3600

    for m in ['mean', 'std', 'max', 'min']:
        new_feats['new_transac_purchase_date_diff_2_sec_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_2_seconds'].agg([m]).values
        new_feats['new_transac_purchase_date_diff_2_day_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_2_days'].agg([m]).values
        new_feats['new_transac_purchase_date_diff_2_hour_{}'.format(m)] = new.groupby(['card_id'])['purchase_date_diff_2_hours'].agg([m]).values

with timer('Special date influence'):
    new['ChristmasDay_2017'] = (pd.to_datetime('2017-12-25') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    new['FathersDay_2017'] = (pd.to_datetime('2017-08-13') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    new['ChildrenDay_2017'] = (pd.to_datetime('2017-10-12') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    new['BlackFriday_2017'] = (pd.to_datetime('2017-11-24') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    new['ValentineDay_2017'] = (pd.to_datetime('2017-06-12') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    new['MothersDay_2018'] = (pd.to_datetime('2018-05-13') - new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).values
    for c in ['ChristmasDay_2017', 'FathersDay_2017', 'ChildrenDay_2017',
              'BlackFriday_2017', 'ValentineDay_2017', 'MothersDay_2018']:
        new_feats['new_transac_{}_mean'.format(c)] = new.groupby(['card_id'])[c].mean().values

with timer('Save new transaction amount features'):
    new_feats.drop(['new_transac_count'], axis=1, inplace=True)
    print('new_feats: {}'.format(new_feats.shape))
    new_feats.to_csv('new_transac_time.csv', index=False)