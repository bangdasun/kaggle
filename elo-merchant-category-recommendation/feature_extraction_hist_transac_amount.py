
import numpy as np
import pandas as pd
from kaggle_learn.utils import timer
from sklearn.decomposition import PCA

with timer('Load data'):
    hist = pd.read_csv('hist_transac_processed.csv')
    print('historical transaction data: {}'.format(hist.shape))

with timer('Get feature dataframe base'):
    hist_feats = pd.DataFrame(hist.groupby(['card_id']).size()).reset_index()
    hist_feats.columns = ['card_id', 'hist_transac_count']

with timer('Transform purchase amount'):
    hist['purchase_amount'] = np.round(hist['purchase_amount'] / 0.00150265118 + 497.06, 2)

with timer('Transaction amount features'):
    for m in ['sum', 'mean', 'max', 'min', 'median', 'std', 'skew']:
        hist_feats['hist_transac_amount_{}'.format(m)] = hist.groupby(['card_id'])['purchase_amount'].agg([m]).values

    hist_feats['hist_transac_amount_diff'] = hist_feats['hist_transac_amount_max'].values - hist_feats['hist_transac_amount_min'].values

with timer('Transaction (time related) features'):
    hist_monthsum_amount = hist.groupby(['card_id', 'month_lag'])['purchase_amount'].sum().unstack().reset_index()
    for i in range(1, 7):
        hist_feats['hist_transac_monthlag_last_{}_amount'.format(i)] = hist_monthsum_amount.iloc[:, -i:].sum(axis=1).values

    for i in range(1, 6):
        for j in range(i + 1, 7):
            hist_feats['hist_transac_monthlag_last_{}_{}_amount_ratio'.format(j, i)] = hist_feats['hist_transac_monthlag_last_{}_amount'.format(j)].values / hist_feats['hist_transac_monthlag_last_{}_amount'.format(i)].values
            hist_feats['hist_transac_monthlag_last_{}_{}_amount_ratio'.format(j, i)] = hist_feats['hist_transac_monthlag_last_{}_{}_amount_ratio'.format(j, i)].replace([np.inf, -np.inf], np.nan)
            hist_feats['hist_transac_monthlag_last_{}_{}_amount_log_ratio'.format(j, i)] = np.log2(hist_feats['hist_transac_monthlag_last_{}_{}_amount_ratio'.format(j, i)])

with timer('Time decay features'):
    hist = hist.sort_values('purchase_date')
    tmp_df = hist.groupby(['card_id']).size().reset_index()
    tmp_df.columns = ['card_id', 'hist_transac_count']
    hist = hist.merge(tmp_df, on=['card_id'], how='left')
    hist['transac_seq_num'] = hist.groupby(['card_id']).cumcount() + 1
    hist['transac_seq_num_desc'] = hist['hist_transac_count'] - hist['transac_seq_num'] - 1
    hist['transac_decay'] = 0.8 ** hist['transac_seq_num_desc'].values
    hist['transac_amount_decay'] = hist['purchase_amount'] * hist['transac_decay']

    hist['transac_month_decay'] = 1.2 ** hist['month_lag'] + 1.
    hist['transac_amount_month_decay'] = hist['purchase_amount'] * hist['transac_month_decay']

    for m in ['sum', 'mean', 'max', 'min', 'median', 'std', 'skew']:
        hist_feats['hist_transac_amount_decay_{}'.format(m)] = hist.groupby(['card_id'])['transac_amount_decay'].agg([m]).values
        hist_feats['hist_transac_amount_month_decay_{}'.format(m)] = hist.groupby(['card_id'])['transac_amount_month_decay'].agg([m]).values

with timer('Transaction amount month sequence decomposition features'):
    num = 5
    pca_decomp = PCA(n_components=num, random_state=4590)
    hist_amt_matrix = hist_monthsum_amount.iloc[:, 1:].fillna(-0.74).values
    pca_decomp_amt = pca_decomp.fit_transform(hist_amt_matrix)
    for i in range(num):
        hist_feats['hist_transac_amount_month_seq_decomp_{}'.format(i)] = pca_decomp_amt[:, i]

with timer('Transaction amount merchant category sequence decomposition features'):
    # hist_merchant_cnt = hist.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].count().unstack().fillna(0.0).reset_index().iloc[:, 1:].values
    hist_merchant_amt = hist.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].sum().unstack().fillna(-3600.0).reset_index().iloc[:, 1:].values
    num = 10
    pca_decomp = PCA(n_components=num, random_state=4590)
    pca_decomp_amt = pca_decomp.fit_transform(hist_merchant_amt)
    for i in range(num):
        hist_feats['hist_transac_amount_merchant_seq_decomp_{}'.format(i)] = pca_decomp_amt[:, i]

# refer from https://www.kaggle.com/fabiendaniel/elo-world?scriptVersionId=8335387
with timer('Higher order transaction amount features'):
    def successive_aggregates(df, field1, field2):
        t = df.groupby(['card_id', field1])[field2].mean()
        u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
        u.columns = ['hist_transac_' + field1 + '_' + field2 + '_' + c for c in u.columns.values]
        u.reset_index(inplace=True)
        return u

    tmp_df_1 = successive_aggregates(hist, 'category_1', 'purchase_amount')
    tmp_df_2 = successive_aggregates(hist, 'installments', 'purchase_amount')
    tmp_df_3 = successive_aggregates(hist, 'city_id', 'purchase_amount')

    tmp_df_4 = successive_aggregates(hist, 'merchant_category_id', 'purchase_amount')
    tmp_df_5 = successive_aggregates(hist, 'merchant_id', 'purchase_amount')
    tmp_df_6 = successive_aggregates(hist, 'subsector_id', 'purchase_amount')
    tmp_df_7 = successive_aggregates(hist, 'category_2', 'purchase_amount')
    tmp_df_8 = successive_aggregates(hist, 'category_3', 'purchase_amount')

    hist_feats = hist_feats.merge(tmp_df_1, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_2, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_3, on=['card_id'], how='left')

    hist_feats = hist_feats.merge(tmp_df_4, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_5, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_6, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_7, on=['card_id'], how='left')
    hist_feats = hist_feats.merge(tmp_df_8, on=['card_id'], how='left')

with timer('Save historical transaction amount features'):
    hist_feats.drop(['hist_transac_count'], axis=1, inplace=True)
    print('hist_feats: {}'.format(hist_feats.shape))
    hist_feats.to_csv('hist_transac_amount.csv', index=False)