
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kaggle_learn.utils import timer
from kaggle_learn.feature_engineering.statistics import *

with timer('Load data'):
    new = pd.read_csv('new_merchant_transactions.csv')

with timer('Get reference date (month)'):
    new['purchase_month'] = new['purchase_date'].astype(str).apply(lambda x: x[:10])
    new['reference_month'] = pd.to_datetime(new['purchase_month']) - new['month_lag'].apply(lambda x: np.timedelta64(x, 'M'))
    new['reference_month'] = new['reference_month'].astype(str).apply(lambda x: x[:7])
    new.drop(['purchase_month'], axis=1, inplace=True)

with timer('Convert categorical to int for new transactions'):
    cols = ['authorized_flag', 'category_1', 'category_3']
    lbl_encoder = LabelEncoder()
    for c in cols:
        new[c] = lbl_encoder.fit_transform(new[c].astype(str))

with timer('Generate simple / intermediate features'):
    new['month_lag=1'] = (new['month_lag'] == 1).astype(int)
    new['month_lag=2'] = (new['month_lag'] == 2).astype(int)

    new['category_2=1'] = (new['category_2'] == 1.).astype(int)
    new['category_2=2'] = (new['category_2'] == 2.).astype(int)
    new['category_2=3'] = (new['category_2'] == 3.).astype(int)
    new['category_2=4'] = (new['category_2'] == 4.).astype(int)
    new['category_2=5'] = (new['category_2'] == 5.).astype(int)

    new['category_3=0'] = (new['category_3'] == 0).astype(int)
    new['category_3=1'] = (new['category_3'] == 1).astype(int)
    new['category_3=2'] = (new['category_3'] == 2).astype(int)
    new['category_3=3'] = (new['category_3'] == 3).astype(int)

    new['purchase_date'] = pd.to_datetime(new['purchase_date'])

with timer('Save processed new transactions'):
    new.to_csv('new_transac_processed.csv', index=False)
