
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from kaggle_learn.utils import timer

# ============================================
# Helper functions
# ============================================

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
    NFOLDS = 5
    oof_train = np.zeros((ntrain, ))
    oof_test = np.zeros((ntest, ))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ============================================
# Run a Ridge to get OOF (out-of-fold)
# ============================================

with open('features.pkl', 'rb') as f:
	features = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
	df_text_processed = pickle.load(f)

with open('df_reduced.pkl', 'rb') as f:
	df_reduced = pickle.load(f)

with timer('Load data'):
	train = pd.read_csv('train.csv.zip', parse_dates = ["activation_date"])
	test = pd.read_csv('test.csv.zip', parse_dates = ["activation_date"])
	ntrain = train.shape[0]
	ntest = test.shape[0]
	del train, test
	gc.collect()

with timer('Training ridge oof preds'):
    y_train_all = df_reduced['deal_probability'].iloc[:ntrain]
        
    if os.path.exists('ridge_preds.csv'):
        ridge_preds = pd.read_csv('ridge_preds.csv')
        df_reduced['ridge_preds'] = ridge_preds['ridge_preds'].values
        del ridge_preds
        gc.collect()
    else:
        kf = KFold(ntrain, n_folds=5, shuffle=True, random_state=42)
        
        ridge_params = {'alpha'        : 30.,
                        'fit_intercept': True,
                        'normalize'    : False,
                        'copy_X'       : True,
                        'max_iter'     : None,
                        'tol'          : .001, 
                        'solver'       : 'auto',
                        'random_state' :42}

        ridge = SklearnWrapper(clf=Ridge, seed=42, params=ridge_params)
        ridge_oof_train, ride_oof_test = get_oof(ridge, df_text_processed[:ntrain], y_train_all, df_text_processed[ntrain:])
        print('Ridge oof rmse = {}'.format(np.sqrt(mean_squared_error(y_train_all, ridge_oof_train))))
        ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
        df_reduced['ridge_preds'] = ridge_preds
        features.append('ridge_preds')

with open('features.pkl', 'wb') as f:
	pickle.dump(features, f)

with open('df_reduced.pkl', 'wb') as f:
	pickle.dump(df_reduced, f)