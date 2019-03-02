
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

with timer('Transaction amount features'):
    for m in ['sum', 'mean', 'max', 'min', 'median', 'std', 'skew']:
        new_feats['new_transac_amount_{}'.format(m)] = new.groupby(['card_id'])['purchase_amount'].agg([m]).values

    new_feats['new_transac_amount_diff'] = new_feats['new_transac_amount_max'].values - new_feats['new_transac_amount_min'].values

with timer('Transaction (time related) features'):
    new_monthsum_amount = new.groupby(['card_id', 'month_lag'])['purchase_amount'].sum().unstack().reset_index()
    new_feats['new_transac_monthlag_last_1_amount'] = new_monthsum_amount.iloc[:, -1].values
    new_feats['new_transac_monthlag_last_2_amount'] = new_monthsum_amount.iloc[:, -2].values
    new_feats['new_transac_monthlag_last_2_1_amount_ratio'] = new_monthsum_amount.iloc[:, -2].values / new_monthsum_amount.iloc[:, -1].values
    new_feats['new_transac_monthlag_last_2_1_amount_ratio'] = new_feats['new_transac_monthlag_last_2_1_amount_ratio'].replace([np.inf, -np.inf], np.nan)
    new_feats['new_transac_monthlag_last_2_1_amount_log_ratio'] = np.log2(new_feats['new_transac_monthlag_last_2_1_amount_ratio'])

with timer('Time decay features'):
    new = new.sort_values('purchase_date')
    tmp_df = new.groupby(['card_id']).size().reset_index()
    tmp_df.columns = ['card_id', 'new_transac_count']
    new = new.merge(tmp_df, on=['card_id'], how='left')
    new['transac_seq_num'] = new.groupby(['card_id']).cumcount() + 1
    new['transac_seq_num_desc'] = new['new_transac_count'] - new['transac_seq_num'] - 1
    new['transac_decay'] = 0.8 ** new['transac_seq_num_desc'].values
    new['transac_amount_decay'] = new['purchase_amount'] * new['transac_decay']

    new['transac_month_decay'] = 1.2 ** new['month_lag'] + 1.
    new['transac_amount_month_decay'] = new['purchase_amount'] * new['transac_month_decay']

    for m in ['sum', 'mean', 'max', 'min', 'median', 'std', 'skew']:
        new_feats['new_transac_amount_decay_{}'.format(m)] = new.groupby(['card_id'])['transac_amount_decay'].agg([m]).values
        new_feats['new_transac_amount_month_decay_{}'.format(m)] = new.groupby(['card_id'])['transac_amount_month_decay'].agg([m]).values

# refer from https://www.kaggle.com/fabiendaniel/elo-world?scriptVersionId=8335387
with timer('Higher order transaction amount features'):
    def successive_aggregates(df, field1, field2):
        t = df.groupby(['card_id', field1])[field2].mean()
        u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
        u.columns = ['new_transac_' + field1 + '_' + field2 + '_' + c for c in u.columns.values]
        u.reset_index(inplace=True)
        return u

    tmp_df_1 = successive_aggregates(new, 'category_1', 'purchase_amount')
    tmp_df_2 = successive_aggregates(new, 'installments', 'purchase_amount')
    tmp_df_3 = successive_aggregates(new, 'city_id', 'purchase_amount')

    tmp_df_4 = successive_aggregates(new, 'merchant_category_id', 'purchase_amount')
    tmp_df_5 = successive_aggregates(new, 'merchant_id', 'purchase_amount')
    tmp_df_6 = successive_aggregates(new, 'subsector_id', 'purchase_amount')
    tmp_df_7 = successive_aggregates(new, 'category_2', 'purchase_amount')
    tmp_df_8 = successive_aggregates(new, 'category_3', 'purchase_amount')

    new_feats = new_feats.merge(tmp_df_1, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_2, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_3, on=['card_id'], how='left')

    new_feats = new_feats.merge(tmp_df_4, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_5, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_6, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_7, on=['card_id'], how='left')
    new_feats = new_feats.merge(tmp_df_8, on=['card_id'], how='left')

with timer('Save new transaction amount features'):
    new_feats.drop(['new_transac_count'], axis=1, inplace=True)
    print('new_feats: {}'.format(new_feats.shape))
    new_feats.to_csv('new_transac_amount.csv', index=False)