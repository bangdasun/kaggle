
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kaggle_learn.utils import timer, reduce_memory_usage
from kaggle_learn.feature_engineering.statistics import *

with timer('Load data'):
    hist = pd.read_csv('historical_transactions.csv')

with timer('Reduce memory usage of historical transaction'):
    hist = reduce_memory_usage(hist)

with timer('Get reference date (month)'):
    hist['purchase_month'] = hist['purchase_date'].astype(str).apply(lambda x: x[:7] + '-28')
    hist['reference_month'] = pd.to_datetime(hist['purchase_month']) - hist['month_lag'].apply(lambda x: np.timedelta64(x, 'M'))
    hist['reference_month'] = hist['reference_month'].astype(str).apply(lambda x: x[:7])
    hist.drop(['purchase_month'], axis=1, inplace=True)

with timer('Convert categorical to int for historical transactions'):
    cols = ['authorized_flag', 'category_1', 'category_3']
    lbl_encoder = LabelEncoder()
    for c in cols:
        hist[c] = lbl_encoder.fit_transform(hist[c].astype(str))

with timer('Generate simple / intermediate features'):
    hist['month_lag=0']  = (hist['month_lag'] == 0).astype(int)
    hist['month_lag=-1'] = (hist['month_lag'] == -1).astype(int)
    hist['month_lag=-2'] = (hist['month_lag'] == -2).astype(int)
    hist['month_lag=-3'] = (hist['month_lag'] == -3).astype(int)
    hist['month_lag=-4'] = (hist['month_lag'] == -4).astype(int)
    hist['month_lag=-5'] = (hist['month_lag'] == -5).astype(int)
    hist['month_lag=-6'] = (hist['month_lag'] == -6).astype(int)

    hist['category_2=1'] = (hist['category_2'] == 1.).astype(int)
    hist['category_2=2'] = (hist['category_2'] == 2.).astype(int)
    hist['category_2=3'] = (hist['category_2'] == 3.).astype(int)
    hist['category_2=4'] = (hist['category_2'] == 4.).astype(int)
    hist['category_2=5'] = (hist['category_2'] == 5.).astype(int)

    hist['category_3=0'] = (hist['category_3'] == 0).astype(int)
    hist['category_3=1'] = (hist['category_3'] == 1).astype(int)
    hist['category_3=2'] = (hist['category_3'] == 2).astype(int)
    hist['category_3=3'] = (hist['category_3'] == 3).astype(int)

    hist['purchase_date'] = pd.to_datetime(hist['purchase_date'])

with timer('Save processed historical transactions'):
    hist.to_csv('hist_transac_processed.csv', index=False)
