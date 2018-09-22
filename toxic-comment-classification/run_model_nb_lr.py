
import re
import time
import string
import numpy as np
import pandas as pd
import gc

from toxic_utils import *
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

english_stemmer = SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse

# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    
    Example
    -------
    >>> NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(training_features, training_labels)
    """
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        # y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p+1) / ((y==y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
		

start_time = time.time()
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# -----------------
# --- read data ---
# -----------------
train = pd.read_csv('../input/train.csv')
train.fillna(' ', inplace=True)
ntrain = train.shape[0]

test = pd.read_csv('../input/test.csv').fillna(' ')
test.fillna(' ', inplace=True)
print('Load data [{}] secs.'.format(time.time() - start_time))

df_full = pd.concat([train, test])
print(df_full.shape)
print(df_full.isnull().sum())


# -----------------------
# --- data processing ---
# -----------------------
X_comment_num_capitals = df_full.comment_text.apply(lambda x: np.sum(1 for c in x if c.isupper())).values.reshape(-1, 1)
X_comment_num_title = df_full.comment_text.apply(lambda x: np.sum(1 for c in x.split() if c.isupper())).values.reshape(-1, 1)

df_full.comment_text = df_full.comment_text.str.lower()
df_full.comment_text = df_full.comment_text.apply(lambda x: clean_comment(x))
df_full.comment_text = df_full.comment_text.apply(lambda x: clean_url(x))
df_full.comment_text = df_full.comment_text.apply(lambda x: character_range(x))


smile = [':d', ':p', ':dd', '8)', ':-)', ':)', ';)', ';-)', '(:', '(-']
X_comment_num_exclamation = df_full.comment_text.apply(lambda x: x.count('!')).values.reshape(-1, 1)
X_comment_num_punct = df_full.comment_text.apply(lambda x: np.sum(x.count(p) for p in string.punctuation)).values.reshape(-1, 1)
X_comment_num_smiles = df_full.comment_text.apply(lambda x: np.sum(x.count(w) for w in smile)).values.reshape(-1, 1)

str_replace(df_full)
print('Clean text finished [{}] secs.'.format(time.time() - start_time))

# word level tfidf
tfidf_vectorizer = StemmedTfidfVectorizer(ngram_range=(1, 2),
                                          min_df=5, 
                                          strip_accents='unicode',
                                          sublinear_tf=True,
                                          max_features=40000,
                                          token_pattern=r'\w{1,}|\!{2,}|\.{2,}')

X_comments = tfidf_vectorizer.fit_transform(df_full.comment_text)
print('Word n_gram tfidf generated [{}] secs.'.format(time.time() - start_time))
del tfidf_vectorizer
gc.collect()

# char level tfidf
tfidf_vectorizer2 = TfidfVectorizer(ngram_range=(1, 6),
                                    sublinear_tf=True,
                                    min_df=3,
                                    strip_accents='unicode',
                                    analyzer='char',
                                    max_features=50000)
X_comments_char = tfidf_vectorizer2.fit_transform(df_full.comment_text)
print('Char n_gram tfidf generated [{}] secs.'.format(time.time() - start_time))

del tfidf_vectorizer2
gc.collect()


X_sparse = hstack((X_comments, X_comments_char)).tocsr()
del X_comments
del X_comments_char
gc.collect()

# --------------------------------------
# --- text statistical meta features --- 
# --------------------------------------
df_full.loc[:, 'comment_len'] = df_full.comment_text.apply(lambda x: len(x))
df_full.loc[:, 'comment_num_words'] = df_full.comment_text.apply(lambda x: len(x.split()))

X_comment_len = df_full.comment_len.values.reshape(-1, 1)
X_comment_len_inv = 1 / (df_full.comment_len.values + 1).reshape(-1, 1)
X_comment_len_log_inv = 1 / np.log1p(df_full.comment_len.values + 1).reshape(-1, 1)
X_comment_num_words = df_full.comment_num_words.values.reshape(-1, 1) # contains 0
X_comment_words_len_ratio = X_comment_num_words * X_comment_len_inv
X_comment_words_len_log_ratio = X_comment_num_words * X_comment_len_log_inv
X_comment_len_inv_pw = X_comment_len_inv ** 1.5
X_comment_capital_len_ratio = X_comment_num_capitals * X_comment_len_inv
X_comment_uniq_words = df_full.comment_text.apply(lambda x: len(set(w for w in x.split()))).values.reshape(-1, 1)
X_comment_uniq_words_ratio = (X_comment_uniq_words + 1) / (X_comment_num_words + 1)
X_comment_exclamation_ratio = X_comment_num_exclamation / (X_comment_num_punct + 1)
X_comment_uniq_words_ratio2 = X_comment_uniq_words * X_comment_len_inv
X_comment_num_sent = df_full.comment_text.apply(lambda x: len(re.split("\\.{1,}|\\!{1,}|\\?{1,}|\n{1,}", x))).values.reshape(-1, 1)
X_comment_avg_sent_len = X_comment_num_sent * X_comment_len_inv
X_comment_uniq_words_sent_ratio = X_comment_uniq_words / X_comment_num_sent
X_comment_title_words_ratio = (X_comment_num_title + 1) / (X_comment_num_words + 1)

print('Numerical features generated [{}] secs.'.format(time.time() - start_time))

X_num_features = np.hstack((X_comment_len_inv,
                            X_comment_len_log_inv,
                            X_comment_num_words,
                            X_comment_words_len_ratio,
                            X_comment_words_len_log_ratio,
                            X_comment_len_inv_pw,
                            X_comment_capital_len_ratio,
                            X_comment_exclamation_ratio,
                            X_comment_uniq_words_ratio,
                            X_comment_num_smiles,
                            X_comment_uniq_words_ratio2,
                            X_comment_avg_sent_len,
                            X_comment_uniq_words_sent_ratio))

del X_comment_len_inv
del X_comment_len_log_inv
del X_comment_num_words
del X_comment_words_len_ratio
del X_comment_words_len_log_ratio
del X_comment_len_inv_pw
del X_comment_capital_len_ratio
del X_comment_exclamation_ratio
del X_comment_uniq_words_ratio
del X_comment_num_smiles
del X_comment_uniq_words_ratio2
del X_comment_avg_sent_len
del X_comment_uniq_words_sent_ratio
gc.collect()

X_train_sparse = X_sparse[:ntrain]
X_test_sparse = X_sparse[ntrain:]
del X_sparse
gc.collect()

X_train_num = X_num_features[:ntrain]
X_test_num = X_num_features[ntrain:]
del X_num_features
gc.collect()

print('Features are all set [{}] secs.'.format(time.time() - start_time))


# -------------------
# --- build model ---
# -------------------
predictions = {'id': test['id']}
C = [1.5, 1.5, 1.5, 1.5, 1.5, .35]

submission = pd.DataFrame.from_dict(predictions)

X_train_all = hstack((X_train_sparse, X_train_num)).tocsr()
X_test_all = hstack((X_test_sparse, X_test_num)).tocsr()

del X_train_sparse, X_train_num
del X_test_sparse, X_test_num
gc.collect()

folds = KFold(n_splits=10, shuffle=True, random_state=233)
losses = []
losses_per_folds = np.zeros(folds.n_splits)
for i_c, class_name in enumerate(class_names):
    class_pred = np.zeros(ntrain)
    train_target = train[class_name]
    submission[class_name] = 0.0
    cv_scores = []
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train_all)):
        clf = NbSvmClassifier(C=C[i_c])
        clf.fit(X_train_all[trn_idx.reshape(-1)], train_target.iloc[trn_idx.reshape(-1)].values)
        class_pred[val_idx.reshape(-1)] = clf.predict_proba(X_train_all[val_idx.reshape(-1)])[:, 1]
        score = roc_auc_score(train_target.iloc[val_idx.reshape(-1)], class_pred[val_idx.reshape(-1)])
        cv_scores.append(score)
        losses_per_folds[n_fold] += score / len(class_names)
    
    cv_score = roc_auc_score(train_target, class_pred)
    losses.append(cv_score)
    train[class_name + '_oof'] = class_pred
    print('CV score for class {} is {}'.format(class_name, cv_score))
    gc.collect()
	
print('Total CV score is {}'.format(np.mean(losses)))
train[['id'] + class_names + [f + '_oof' for f in class_names]].to_csv('nb_lr_oof_preds.csv', index=False)

for i_c, class_name in enumerate(class_names):
    train_target = train[class_name]
    model = NbSvmClassifier(C=C[i_c])
    model.fit(X_train_all, train_target)
    submission[class_name] = model.predict_proba(X_test_all)[:, 1]
    
submission.to_csv('nb_lr_sub_preds.csv', index=False)