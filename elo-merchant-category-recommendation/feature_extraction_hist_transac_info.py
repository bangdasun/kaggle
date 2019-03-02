
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer

def longest_streak(arr):
    """
    https://codereview.stackexchange.com/questions/138550/count-consecutive-ones-in-a-binary-list
    """
    one_list = []
    size = 0
    for num in arr:
        if num == 1:
            one_list.append(num)
        elif num == 0 and size < len(one_list):
            size = len(one_list)
            one_list = []
    return max(size, len(one_list))

def group_entropy(df, group, subgroup, cname, value, df_feats):
    if isinstance(subgroup, list):
        full_group = [group]
        full_group.extend(subgroup)
    else:
        full_group = [group, subgroup]

    gp_1 = df.groupby(full_group)[value].count().reset_index()
    gp_1.columns = full_group + ['subgroup_cnt']

    gp_2 = df.groupby(group)[value].count().reset_index()
    gp_2.columns = [group, 'cnt']

    gp_3 = gp_2.merge(gp_1, on=group, how='left')

    gp_3['entropy'] = -np.log(gp_3['subgroup_cnt'] / gp_3['cnt']) * gp_3['subgroup_cnt'] / gp_3['cnt']
    gp_3['entropy'].fillna(0, inplace=True)
    gp_4 = gp_3.groupby(group)['entropy'].sum().reset_index()
    gp_4.columns = [group, cname]
    df_feats = df_feats.merge(gp_4, on=group, how='left')
    return df_feats

with timer('Load data'):
    hist = pd.read_csv('hist_transac_processed.csv')
    print('historical transaction data: {}'.format(hist.shape))

with timer('Get feature dataframe base'):
    hist_feats = pd.DataFrame(hist.groupby(['card_id']).size()).reset_index()
    hist_feats.columns = ['card_id', 'hist_transac_count']

with timer('Basic transaction info features'):
    for c in ['city', 'state', 'merchant_category', 'subsector', 'merchant']:
        hist_feats['hist_transac_{}_nunique'.format(c)] = hist.groupby(['card_id'])['{}_id'.format(c)].nunique().values

    hist_feats['hist_transac_category_1_1_count'] = hist.groupby(['card_id'])['category_1'].sum().values
    hist_feats['hist_transac_category_1_0_count'] = hist_feats['hist_transac_count'].values - hist_feats['hist_transac_category_1_1_count'].values
    hist_feats['hist_transac_category_1_1_mean'] = hist.groupby(['card_id'])['category_1'].mean().values
    hist_feats['hist_transac_category_1_1_std'] = hist.groupby(['card_id'])['category_1'].std().values

    for c in ['category_2=1', 'category_2=2', 'category_2=3', 'category_2=4', 'category_2=5',
              'category_3=0', 'category_3=1', 'category_3=2', 'category_3=3']:
        hist_feats['hist_transac_{}_count'.format(c)] = hist.groupby(['card_id'])[c].sum().values
        hist_feats['hist_transac_{}_mean'.format(c)]  = hist.groupby(['card_id'])[c].mean().values

    for m in ['mean', 'sum', 'max', 'min', 'std', 'skew']:
        hist_feats['hist_transac_installments_{}'.format(m)] = hist.groupby(['card_id'])['installments'].agg([m]).values

    hist_feats['hist_transac_approved_count'] = hist.groupby(['card_id'])['authorized_flag'].sum().values
    hist_feats['hist_transac_approved_mean'] = hist.groupby(['card_id'])['authorized_flag'].mean().values
    hist_feats['hist_transac_denied_count'] = hist_feats['hist_transac_count'].values - hist_feats['hist_transac_approved_count'].values

with timer('Monthly transaction info features'):
    hist_monthsum_count = hist.groupby(['card_id', 'month_lag'])['purchase_amount'].count().unstack().fillna(0.0).reset_index()
    hist_feats['hist_transac_monthlag_count_std'] = hist_monthsum_count.iloc[:, 1:].std(axis=1).values
    hist_feats['hist_transac_monthlag_count_max'] = hist_monthsum_count.iloc[:, 1:].max(axis=1).values
    hist_have_purchase = (hist_monthsum_count.iloc[:, 1:] != 0).astype(int).values
    hist_feats['hist_transac_monthlag_streak_max'] = np.apply_along_axis(longest_streak, 1, hist_have_purchase)
    
with timer('Popular merchants features'):
    hist_feats['hist_transac_merchant_count_mean'] = hist_feats['hist_transac_count'].values / hist_feats['hist_transac_merchant_nunique'].values
    hist_feats['hist_transac_merchant_count_max'] = hist.groupby(['card_id', 'merchant_id']).size().reset_index().groupby(['card_id'])[0].max().values
    hist_feats['hist_transac_merchant_count_max_sum_ratio'] = hist_feats['hist_transac_merchant_count_max'].values / hist_feats['hist_transac_count'].values
    hist_feats['hist_transac_merchant_count_max_mean_ratio'] = hist_feats['hist_transac_merchant_count_max'].values / hist_feats['hist_transac_merchant_count_mean'].values
    hist_feats['hist_transac_merchant_count_std'] = hist.groupby(['card_id', 'merchant_id']).size().reset_index().groupby(['card_id'])[0].std().values

    tmp_df = hist.groupby(['card_id', 'merchant_id']).size().reset_index()
    tmp_df.columns = ['card_id', 'merchant_id', 'merchant_count']
    for i in [1, 2, 5]:
        tmp_df['revisited>{}'.format(i)] = (tmp_df['merchant_count'] > i).astype(int)
        hist_feats['hist_transac_merchant_visited>{}'.format(i)] = tmp_df.groupby(['card_id'])['revisited>{}'.format(i)].mean().values

    for m in ['std', 'skew']:
        hist_feats['hist_transac_merchant_visit_count_{}'.format(m)] = tmp_df.groupby(['card_id'])['merchant_count'].agg([m]).values

with timer('Group entropy features'):
    for c in ['merchant_category_id', 'subsector_id', 'merchant_id', 'city_id', 'state_id',
              'category_1', 'category_2', 'category_3', 'month_lag']:
        hist_feats = group_entropy(hist, 'card_id', c, 'hist_transac_{}_entropy'.format(c), 'purchase_amount', hist_feats)

with timer('Save historical transaction info features'):
    print('hist_feats: {}'.format(hist_feats.shape))
    hist_feats.to_csv('hist_transac_info.csv', index=False)