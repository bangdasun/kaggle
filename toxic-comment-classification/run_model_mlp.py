
# fork this kernel from
# https://www.kaggle.com/johnfarrell/tfidf-3layers-mlp-from-mercari

import os; os.environ['OMP_NUM_THREADS'] = '4'
import time
import re
import string
import numpy as np
import pandas as pd
import keras as ks
import tensorflow as tf

from toxic_utils import *
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
from typing import List, Dict

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
    

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.comment_text = df.comment_text.str.lower()
    df.comment_text = df.comment_text.apply(lambda x: clean_comment(x))
    df.comment_text = df.comment_text.apply(lambda x: clean_url(x))
    df.comment_text = df.comment_text.apply(lambda x: character_range(x))
    str_replace(df)
    return df[['comment_text']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_valid_predict(xs, ys) -> np.ndarray:
    X_train, X_valid, X_test = xs
    y_train, y_valid = ys
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
        
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_valid_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(512, activation='relu')(model_in)
        out = ks.layers.Dropout(0.3)(out)
        out = ks.layers.Dense(6, activation="sigmoid")(out)
        model = ks.Model(model_in, out)
        lr_init = 1e-3
        model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=lr_init, decay=0.000015))
        cv_score = -1
        pred_valid_best = None
        best_model = None
        
        for i in range(2):
            if i < 7:
                batch_size = 2 ** (5 + i)
            with timer(f'epoch {i + 1}'):
                ks.backend.set_value(model.optimizer.lr, lr_init * 0.7 ** i)
                model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1, verbose=0)
                pred_valid=model.predict(X_valid)
                cv_i = np.mean(roc_auc_score(y_valid, pred_valid, average=None))
                if cv_i > cv_score: 
                    cv_score = cv_i
                    print('best_score', cv_score, '@', f'epoch {i + 1}')
                    pred_valid_best = pred_valid.copy()
                    best_model = ks.models.clone_model(model)
                    best_model.set_weights(model.get_weights())
                    
        res = dict(pred_valid=pred_valid_best, 
                   pred_test=best_model.predict(X_test))
        return res

class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# from https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

word_vectorizer = StemmedTfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df=10,
    token_pattern=r'\w{1,}|\!{2,}|\.{2,}',
    max_features=40000,
    ngram_range=(1, 2))
    
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 6),
    max_features=50000) # ?
    
vectorizer = make_union(
    on_field('comment_text', word_vectorizer),
    on_field('comment_text', char_vectorizer),
    n_jobs=4)
    
    
train = pd.read_csv('../input/train.csv')
ntrain = train.shape[0]

test = pd.read_csv('../input/test.csv')

df_full = pd.concat([train, test])

with timer('process train'):
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]
    
    y_train = train[class_names].values
    y_valid = valid[class_names].values
    X_train = vectorizer.fit(preprocess(df_full)).transform(preprocess(train))
    print(f'X_train: {X_train.shape} of {X_train.dtype}')
    del train
    gc.collect()
    
with timer('process valid'):
    X_valid =vectorizer.transform(preprocess(valid))
    del valid
    gc.collect()
    
with timer('process test'):
    X_test = vectorizer.transform(preprocess(test))
    del test
    gc.collect()
    
with ThreadPool(processes=4) as pool:
    Xb_train, Xb_valid, Xb_test = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid, X_test]]
    xs = [[Xb_train, Xb_valid, Xb_test], [X_train, X_valid, X_test]]
    ys = [y_train, y_valid]
    res_li = pool.map(partial(fit_valid_predict, ys=ys), xs)
    pred_valid_avg = np.mean([res['pred_valid'] for res in res_li], axis=0)
    pred_test_avg = np.mean([res['pred_test'] for res in res_li], axis=0)
    auc_valid = np.mean(roc_auc_score(y_valid, pred_valid_avg, average=None))
    
print('Valid auc: {:.6f}'.format(auc_valid))

sub = pd.read_csv('../input/sample_submission.csv')
sub[class_names] = pred_test_avg.copy()
sub.to_csv('mlp_sub_preds.csv'.format(auc_valid), index=False)
