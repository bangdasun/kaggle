
# This is my private kernel run on kaggle platform
import re
import gc
import time
import pickle
import string
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

NFOLDS = 5
SEED = 42
VALID = True

start_time = time.time()

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

# from https://www.kaggle.com/demery/lightgbm-with-ridge-feature
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

def process_description(df):
    df['description'] = df['description'].fillna('') + ' ' + \
                        df['parent_category_name'].fillna('') + ' ' + \
                        df['category_name'].fillna('')
    
    df['title'] = df['title'] + ' ' + \
                  df['param_1'].fillna('') + ' ' + \
                  df['param_2'].fillna('') + ' ' + \
                  df['param_3'].fillna('') + ' ' + \
                  df['city'].fillna('') + ' ' + \
                  df['region'].fillna('') + ' ' + \
                  df['user_type'].fillna('') + ' ' + \
                  df['image_top_1'].fillna('').map(str).apply(lambda x: 'img_class' + x)
                  
    return df['title'], df['description']

usecols = ['title', 'description', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3',
            'city', 'region', 'user_type', 'image_top_1']
            
train = pd.read_csv('../input/train.csv', usecols=usecols + ['deal_probability'])
traindex = train.index
test = pd.read_csv('../input/test.csv', usecols=usecols)
testdex = test.index

print('[{:.2f}] Load data'.format(time.time() - start_time))

ntrain = train.shape[0]
ntest = test.shape[0]

train['title'], train['description'] = process_description(train)
test['title'], test['description'] = process_description(test)

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = train.deal_probability.copy()
train.drop("deal_probability",axis=1, inplace=True)

df = pd.concat([train, test], axis=0)
del train, test
gc.collect()

russian_stop = set(stopwords.words('russian'))
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "smooth_idf": False
}

vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=300000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            preprocessor=get_col('title')))
    ])

print()
vectorizer.fit(df.to_dict('records'))
print('[{:.2f}] Tfidf fitted'.format(time.time() - start_time))

sparse_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

svm_params = {'max_iter': 1500, 'C': 0.01, 'random_state': 42, 'loss': 'squared_epsilon_insensitive'}
ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
                
svm = SklearnWrapper(clf=Ridge, seed=42, params=ridge_params)
svm_oof_train, svm_oof_test = get_oof(svm, sparse_df[:ntrain], y, sparse_df[ntrain:])
svm_preds = np.concatenate([svm_oof_train, svm_oof_test])

with open('svr_preds.pkl', 'wb') as f:
    pickle.dump(svm_preds, f)