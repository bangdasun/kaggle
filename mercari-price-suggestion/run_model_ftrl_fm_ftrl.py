# fork from https://www.kaggle.com/tunguz/wordbatch-ftrl-fm-lgb-lbl-0-42506

import os
import gc
import re
import time
import pickle
import logging
import numpy as np
import pandas as pd

import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

NUM_BRANDS = 4800
NUM_CATEGORIES = 1290
punct_split = r',|.|!|?|~|;|*|:||'
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('sub_fm_fm_ftrl_{}.log'.format(logger_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
    
def split_category(category_name):
    try:
        return category_name.split('/')
    except:
        return 'no label', 'no label', 'no label'

def cut_df(df):
    pop_brand = df['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    df.loc[~df['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    
    pop_cat_1 = df['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_cat_2 = df['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_cat_3 = df['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    df.loc[~df['general_cat'].isin(pop_cat_1), 'general_cat'] = 'missing'
    df.loc[~df['subcat_1'].isin(pop_cat_2), 'subcat_1'] = 'missing'
    df.loc[~df['subcat_2'].isin(pop_cat_3), 'subcat_2'] = 'missing'
    return df

def impute_missing_value(df):
    for c in ['general_cat', 'subcat_1', 'subcat_2', 'brand_name',
              'item_description']:
        df[c] = df[c].fillna(value='missing', inplace=True)
    return df
    
def to_categorical(df):
    df['general_cat'] = df['general_cat'].astype('category')
    df['subcat_1'] = df['subcat_1'].astype('category')
    df['subcat_2'] = df['subcat_2'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')
    return df
    
def normalize_text(text):
    return  u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])
         
def main(test, logger):
    
    logger.info('Start . . .')
    train = pd.read_table('../input/train.tsv', engine='c')
    logger.info('Load train')
    logger.info('train shape {}'.format(train.shape))
    logger.info('test shape {}'.format(test.shape))
    nrow_train = train.shape[0]
    y = np.log1p(train['price'])
    
    train_low_price = train.loc[train['price'] < 1.]
    train = train.drop(train[train['price'] < 1.].index)
    del train_low_price['price']
    logger.info('train_low_price shape {}'.format(train_low_price.shape))
    
    df_full = pd.concat([train, train_low_price, test])
    logger.info('df_full shape {}'.format(df_full.shape))
    
    sub = test[['test_id']]
    logger.info('sub shape {}'.format(sub.shape))
    
    del train, test
    gc.collect()
    
    df_full['general_cat'], df_full['subcat_1'], df_full['subcat_2'] = zip(*df_full['category_name'].apply(lambda x: split_category(x)))
    df_full.drop(['category_name'], axis=1, inplace=True)
    logger.info('Split category_name')
    gc.collect()
    
    df_full = impute_missing_value(df_full)
    logger.info('Impute missing value')
    gc.collect()
    
    df_full = cut_df(df_full)
    logger.info('Cut categories')
    gc.collect()
    
    df_full = to_categorical(df_full)
    logger.info('Convert to categorical features')
    gc.collect()
    
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(df_full['name'])
    del wb
    gc.collect()
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    logger.info('Vectorize name')
    gc.collect()
    
    cnt_vec = CountVectorizer()
    X_cat_1 = cnt_vec.fit_transform(df_full['general_cat'])
    X_cat_2 = cnt_vec.fit_transform(df_full['subcat_1'])
    X_cat_3 = cnt_vec.fit_transform(df_full['subcat_2'])
    df_full.drop(['general_cat', 'subcat_1', 'subcat_2'], axis=1, inplace=True)
    del cnt_vec
    gc.collect()
    logger.info('Vectorize category (general_cat, subcat_1, subcat_2)')
    
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}), procs=2)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(df_full['item_description'])
    del wb
    gc.collect()
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    logger.info('Vectorize item_description')
    gc.collect()
    
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(df_full['brand_name'])
    df_full.drop(['brand_name'], axis=1, inplace=True)
    del lb
    gc.collect()
    logger.info('Label binarize brand_name')
    
    X_dummies = csr_matrix(pd.get_dummies(df_full[['item_condition_id', 'shipping']], sparse=True).values)
    df_full.drop(['item_condition_id', 'shipping'], axis=1, inplace=True)
    logger.info('Get dummies on item_condition_id and shipping')
    gc.collect()
    
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_cat_1, X_cat_2, X_cat_3, X_name)).tocsr()
    logger.info('Create sparse features')
    logger.info('sparse_merge shape {}'.format(sparse_merge.shape))
    del X_dummies, X_description, X_brand, X_cat_1, X_cat_2, X_cat_3, X_name
    gc.collect()
    
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    logger.info('Remove features with doc frequency <= 1')
    logger.info('sparse_merge shape {}'.format(sparse_merge.shape))
    
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    sparse_merge_shape = sparse_merge.shape
    del sparse_merge
    gc.collect()
    
    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge_shape[1],
                 iters=30, inv_link="identity", threads=1)
    model.fit(X, y)
    logger.info('Fit FTRL')
    preds_FTRL = model.predict(X_test)
    logger.info('Predict FTRL')
                 
    model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=sparse_merge_shape[1],
                    alpha_fm=0.01, L2_fm=0.0, init_fm=0.01, D_fm=200, e_noise=0.0001,
                    iters=20, inv_link="identity", threads=4)
    model.fit(X, y)
    logger.info('Fit FM_FTRL')
    preds_FM_FTRL = model.predict(X_test)
    logger.info('Predict FM_FTRL')
    
    preds = (np.expm1(preds_FTRL) * 0.15 + np.expm1(preds_FM_FTRL) * 0.85)
    logger.info('Final predictions generated')
    return preds
    
if __name__ == '__main__':
    logger_main = get_logger('main')
    logger_main.info('Start . . .')
    TEST_CHUNK_SIZE = 700000
    
    def load_test():
        for df in pd.read_table('../input/test_stg2.tsv', chunksize=TEST_CHUNK_SIZE):
            yield df
    
    preds = []
    for idx, df_test in enumerate(load_test()):
        logger_main.info('Generating batch {}'.format(idx))
        logger = get_logger('mercari_main_{}'.format(idx))
        batch_preds = main(df_test, logger)
        batch_preds_list = batch_preds.flatten().tolist()
        
        with open('batch_preds_{}.pkl'.format(idx), 'wb') as f:
            pickle.dump(batch_preds_list, f)
            
        preds.extend(batch_preds_list)
    
    logger_main.info('Generating submission')
    sub = pd.read_table('../input/test_stg2.tsv', engine='c', usecols=['test_id'])
    sub['price'] = np.array(preds).reshape(-1, 1)
    sub.to_csv('sub_fm_fm_ftrl.csv', index=False)