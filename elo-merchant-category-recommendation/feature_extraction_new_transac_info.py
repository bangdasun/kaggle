
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer

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
    new = pd.read_csv('new_transac_processed.csv')
    print('new transaction data: {}'.format(new.shape))

with timer('Get feature dataframe base'):
    new_feats = pd.DataFrame(new.groupby(['card_id']).size()).reset_index()
    new_feats.columns = ['card_id', 'new_transac_count']

with timer('Basic new transaction features'):
    for c in ['city', 'state', 'merchant_category', 'subsector', 'merchant']:
        new_feats['new_transac_{}_nunique'.format(c)] = new.groupby(['card_id'])['{}_id'.format(c)].nunique().values

    new_feats['new_transac_category_1_1_count'] = new.groupby(['card_id'])['category_1'].sum().values
    new_feats['new_transac_category_1_0_count'] = new_feats['new_transac_count'].values - new_feats['new_transac_category_1_1_count'].values
    new_feats['new_transac_category_1_1_mean'] = new.groupby(['card_id'])['category_1'].mean().values
    new_feats['new_transac_category_1_1_std'] = new.groupby(['card_id'])['category_1'].std().values

    for c in ['category_2=1', 'category_2=2', 'category_2=3', 'category_2=4', 'category_2=5',
              'category_3=0', 'category_3=1', 'category_3=2', 'category_3=3']:
        new_feats['new_transac_{}_count'.format(c)] = new.groupby(['card_id'])[c].sum().values
        new_feats['new_transac_{}_mean'.format(c)]  = new.groupby(['card_id'])[c].mean().values

    for m in ['mean', 'sum', 'max', 'min', 'std', 'skew']:
        new_feats['new_transac_installments_{}'.format(m)] = new.groupby(['card_id'])['installments'].agg([m]).values

with timer('Monthly transaction info features'):
    new_monthsum_count = new.groupby(['card_id', 'month_lag'])['purchase_amount'].count().unstack().fillna(0.0).reset_index()
    new_feats['new_transac_monthlag_count_std'] = new_monthsum_count.iloc[:, 1:].std(axis=1).values
    new_feats['new_transac_monthlag_count_max'] = new_monthsum_count.iloc[:, 1:].max(axis=1).values

with timer('Save new transaction info features'):
    print('new_feats: {}'.format(new_feats.shape))
    new_feats.to_csv('new_transac_info.csv', index=False)
