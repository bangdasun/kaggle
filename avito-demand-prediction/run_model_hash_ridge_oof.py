
# This is my private kernel run on kaggle platform
# Reference: Serg @address: https://www.kaggle.com/rumbok, kernel: https://www.kaggle.com/rumbok/ridge-lb-0-41944
import os
import gc
import re
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from time import time
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from nltk.corpus import stopwords 
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge
from sklearn import preprocessing

INPUT_PATH = r'../input'


NFOLDS = 5
SEED = 42
VALID = True

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    oof_rmse = []
    for i, (train_index, test_index) in enumerate(kf):
        print('========== Fold {} =========='.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        fold_rmse = rmse(y[test_index], oof_train[test_index])
        oof_rmse.append(fold_rmse)
        print('RMSE = {}'.format(fold_rmse))
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    print('OOF RMSE = {}'.format(np.mean(oof_rmse)))
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def get_col(col_name): 
	return lambda x: x[col_name]

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, dataframe):
        print('[{:.2f}] select {}'.format(time() - self.start_time, self.field))
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]

class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df
    
    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self
    
    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]
    
def process_data(train, test, start_time=time()):
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train['deal_probability'].values
    merge = pd.concat([train, test])
    
    del train
    del test
    gc.collect()
    
    merge['parent_category_user'] = merge['parent_category_name'].map(str) + '_' + merge['user_type'].apply(str)
    merge['category_user'] = merge['category_name'].map(str) + '_' + merge['user_type'].map(str)
    merge['parent_category_img'] = merge['parent_category_name'].map(str) + '_' + merge['image_top_1'].apply(str)
    merge['category_img'] = merge['category_name'].map(str) + '_' + merge['image_top_1'].map(str)
    print('[{:.2f}] categories concatenated'.format(time() - start_time))
    
    merge['image_top_1'] = merge['image_top_1'].fillna(-1).astype('category')
    
    merge['description'] = merge['description'].fillna('').str.lower()
    merge['param_1'] = merge['param_1'].fillna('').str.lower()
    merge['param_2'] = merge['param_2'].fillna('').str.lower()
    merge['param_3'] = merge['param_3'].fillna('').str.lower()
    merge['description'] = merge['description'] \
                            + ' ' + merge['title'] \
                            + ' ' + merge['param_1'] \
                            + ' ' + merge['param_2'] \
                            + ' ' + merge['param_3']
    print('[{:.2f}] description concatenated'.format(time() - start_time))
    return merge, y_train, ntrain, ntest

russian_stop = frozenset(['а', 'без', 'более', 'больше', 'будет', 'будто', 'бы', 'был', 'была', 'были', 'было', 'быть',
	'в', 'вам', 'вас', 'вдруг', 'ведь', 'во', 'вот', 'впрочем', 'все', 'всегда',
	'всего', 'всех', 'всю', 'вы', 'где', 'да', 'даже', 'два', 'для', 'до', 'другой', 'его', 'ее', 'ей',
	'ему', 'если', 'есть', 'еще', 'ж', 'же', 'за', 'зачем', 'здесь', 'и', 'из', 'или', 'им', 'иногда',
	'их', 'к', 'как', 'какая', 'какой', 'когда', 'конечно', 'кто', 'куда', 'ли', 'лучше', 'между', 'меня', 'мне', 'много',
	'может', 'можно', 'мой', 'моя', 'мы', 'на', 'над', 'надо', 'наконец', 'нас', 'не', 'него', 'нее',
	'ней', 'нельзя', 'нет', 'ни', 'нибудь', 'никогда', 'ним', 'них', 'ничего', 'но', 'ну',
	'о', 'об', 'один', 'он', 'она', 'они', 'опять', 'от', 'перед', 'по', 'под', 'после',
	'потом', 'потому', 'почти', 'при', 'про', 'раз', 'разве', 'с', 'сам', 'свою',
	'себе', 'себя', 'сейчас', 'со', 'совсем', 'так', 'такой', 'там', 'тебя', 'тем',
	'теперь', 'то', 'тогда', 'того', 'тоже', 'только', 'том', 'тот', 'три', 'тут',
	'ты', 'у', 'уж', 'уже', 'хорошо', 'хоть', 'чего', 'чем', 'через', 'что',
	'чтоб', 'чтобы', 'чуть', 'эти', 'этого', 'этой', 'этом', 'этот', 'эту', 'я'])

start_time = time()
vectorizer = FeatureUnion([
    ('parent_category_user', Pipeline([
        ('select', ItemSelector('parent_category_user', start_time=start_time)),
        ('transform', CountVectorizer(
            min_df=2,
            lowercase=False
        )),
    ])),
    ('category_user', Pipeline([
        ('select', ItemSelector('category_user', start_time=start_time)),
        ('transform', CountVectorizer(
            min_df=2,
            lowercase=False
        )),
    ])),
    ('parent_category_img', Pipeline([
        ('select', ItemSelector('parent_category_img', start_time=start_time)),
        ('transform', CountVectorizer(
            min_df=2,
            lowercase=False
        )),
    ])),
    ('category_img', Pipeline([
        ('select', ItemSelector('category_img', start_time=start_time)),
        ('transform', CountVectorizer(
            min_df=2,
            lowercase=False
        )),
    ])),
    ('image_top_1', Pipeline([
        ('select', ItemSelector('image_top_1', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('description', Pipeline([
        ('select', ItemSelector('description', start_time=start_time)),
        ('hash', HashingVectorizer(
            ngram_range=(1, 3),
            n_features=2 ** 27,
            dtype=np.float32,
            norm='l2',
            lowercase=False,
            stop_words=russian_stop
        )),
        ('drop_cols', DropColumnsByDf(min_df=2)),
    ]))
], n_jobs=1)

start_time = time()
train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))
merge, y_train, ntrain, ntest = process_data(train, test, start_time)
sparse_merge = vectorizer.fit_transform(merge)

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
ridge = SklearnWrapper(clf=Ridge, seed=42, params=ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, sparse_merge[:ntrain], y_train, sparse_merge[ntrain:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

with open('hash_ridge_preds.pkl', 'wb') as f:
    pickle.dump(ridge_preds, f)


