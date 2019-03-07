# fork from https://www.kaggle.com/rumbok/ridge-lb-0-41944

import os
import gc
import re
import pickle
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

from time import time
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from nltk.corpus import stopwords

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
INPUT_PATH = r'../input'

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('sub_ridge_{}.log'.format(logger_name))
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
        cats = category_name.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'    
        
def dameraulevenshtein(seq1, seq2):
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
                
    return thisrow[len(seq2) - 1]

class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None
            
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time()-self.start_time:.3f}] select {self.field}')
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
        
def brands_filling(df, logger):
    vc = df['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+\s[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    logger.info(f"Before empty brand_name: {len(df[df['brand_name'] == ''].index)}")

    n_name = df[df['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+\s[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+")
    df.loc[df['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = df[df['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+\s[a-z0-9*/+\-'§¡??!.,|&%???§Ö§Ö§Ö§Õ§Ö¨g]+")
    df.loc[df['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = df[df['brand_name'] == '']['name'].str.findall(pat=brand_word)
    df.loc[df['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = df[df['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    df.loc[df['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    logger.info(f"After empty brand_name: {len(df[df['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()

def preprocess_regex(df, logger, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    df['name'] = df['name'].str.replace(karats_regex, karats_repl)
    df['item_description'] = df['item_description'].str.replace(karats_regex, karats_repl)
    logger.info(f'[{time() - start_time:.3f}] Karats normalized.')

    df['name'] = df['name'].str.replace(unit_regex, unit_repl)
    df['item_description'] = df['item_description'].str.replace(unit_regex, unit_repl)
    logger.info(f'[{time() - start_time:.3f}] Units glued.')
 
def preprocess_pandas(train, test, logger, start_time=time()):
    train = train[train.price > 0.0].reset_index(drop=True)
    logger.info('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    df_full: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    df_full['has_category'] = (df_full['category_name'].notnull()).astype('category')
    logger.info(f'[{time() - start_time:.3f}] Has_category filled.')

    df_full['category_name'] = df_full['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    df_full['general_cat'], df_full['subcat_1'], df_full['subcat_2'], df_full['gen_subcat1'] = \
        zip(*df_full['category_name'].apply(lambda x: split_category(x)))
    logger.info(f'[{time() - start_time:.3f}] Split categories completed.')

    df_full['has_brand'] = (df_full['brand_name'].notnull()).astype('category')
    logger.info(f'[{time() - start_time:.3f}] Has_brand filled.')

    df_full['gencat_cond'] = df_full['general_cat'].map(str) + '_' + df_full['item_condition_id'].astype(str)
    df_full['subcat_1_cond'] = df_full['subcat_1'].map(str) + '_' + df_full['item_condition_id'].astype(str)
    df_full['subcat_2_cond'] = df_full['subcat_2'].map(str) + '_' + df_full['item_condition_id'].astype(str)
    logger.info(f'[{time() - start_time:.3f}] Categories and item_condition_id concancenated.')

    df_full['name'] = df_full['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    df_full['brand_name'] = df_full['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    df_full['item_description'] = df_full['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    logger.info(f'[{time() - start_time:.3f}] Missing filled.')

    preprocess_regex(df_full, logger, start_time)

    brands_filling(df_full, logger)
    logger.info(f'[{time() - start_time:.3f}] Brand name filled.')

    df_full['name'] = df_full['name'] + ' ' + df_full['brand_name']
    logger.info(f'[{time() - start_time:.3f}] Name concancenated.')

    df_full['item_description'] = df_full['item_description'] \
                                + ' ' + df_full['name'] \
                                + ' ' + df_full['subcat_1'] \
                                + ' ' + df_full['subcat_2'] \
                                + ' ' + df_full['general_cat'] \
                                + ' ' + df_full['brand_name']
    logger.info(f'[{time() - start_time:.3f}] Item description concatenated.')

    df_full.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return df_full, y_train, nrow_train

def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res
    
def main(test, logger):
    mp.set_start_method('forkserver', True)
    start_time = time()
    train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                          engine='c',
                          dtype={'item_condition_id': 'category',
                                 'shipping': 'category'})
    logger.info(f'[{time() - start_time:.3f}] Finished to load data')
    logger.info('Train shape: ', train.shape)
    logger.info('Test shape: ', test.shape)
    
    merge, y_train, nrow_train = preprocess_pandas(train, test, logger, start_time)
    meta_params = {'name_ngram': (1, 2),
                  'name_max_f': 75000,
                  'name_min_df': 10,

                  'category_ngram': (2, 3),
                  'category_token': '.+',
                  'category_min_df': 10,

                  'brand_min_df': 10,

                  'desc_ngram': (1, 3),
                  'desc_max_f': 150000,
                  'desc_max_df': 0.5,
                  'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
    
    vectorizer = FeatureUnion([
        # pipeline: processing name
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        
        # pipeline: process category_name
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_category,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        
        # pipeline: process brand_name
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        
        # pipeline: process condition
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        
        # pipeline: process subcategory_1 + condition
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        
        # pipeline: process subcategory_2 + condition
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        
        # pipeline: process has_brand
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        
        # pipeline: process shipping
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        
        # pipeline: process condition
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        
        # pipeline: item_description
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    sparse_merge = vectorizer.fit_transform(merge)
    logger.info(f'[{time() - start_time:.3f}] Merge vectorized')
    logger.info(f'[sparse_merge shape {sparse_merge.shape}]')

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(sparse_merge)
    logger.info(f'[{time() - start_time:.3f}] TF/IDF completed')
    logger.info(f'[{time() - start_time:.3f}] Add new features')

    X_train = X[:nrow_train]
    logger.info(f'[X_train shape {X_train.shape}]')

    X_test = X[nrow_train:]
    del merge
    del sparse_merge
    del vectorizer
    del tfidf_transformer
    gc.collect()

    X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
    logger.info(f'[{time() - start_time:.3f}] Drop only in train or test cols: {X_train.shape[1]}')
    gc.collect()

    ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    ridge.fit(X_train, y_train)
    logger.info(f'[{time() - start_time:.3f}] Train Ridge completed. Iterations: {ridge.n_iter_}')

    predsR = ridge.predict(X_test)
    logger.info(f'[{time() - start_time:.3f}] Predict Ridge completed.')
    
    return np.expm1(predsR)
    
if __name__ == '__main__':
    logger_main = get_logger('main')
    logger_main.info('Start . . .')
    TEST_CHUNK_SIZE = 700000
    
    def load_test():
        for df in pd.read_table('../input/test_stg2.tsv',  dtype={'item_condition_id': 'category','shipping': 'category'}, chunksize=TEST_CHUNK_SIZE):
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
    sub.to_csv('sub_ridge.csv', index=False)
