
# fork this kernel from 
# https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram
# Total CV score is 0.9829690775167856
import os
import gc
import re
import sys
import time
import string
import psutil
import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict


# Contraction replacement patterns
cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


from nltk.stem import SnowballStemmer

english_stemmer = SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))
    
    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))

def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


if __name__ == '__main__':
    gc.enable()
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    with timer("Reading input files"):
        train = pd.read_csv('../input/train.csv').fillna(' ')
        test = pd.read_csv('../input/test.csv').fillna(' ')

    with timer("Performing basic NLP"):
        get_indicators_and_clean_comments(train)
        get_indicators_and_clean_comments(test)


    # Scaling numerical features with MinMaxScaler though tree boosters don't need that
    with timer("Creating numerical features"):
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + class_names]

        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
        test_num_features = csr_matrix(skl.fit_transform(test[num_features]))

    # Get TF-IDF features
    train_text = train['clean_comment']
    test_text = test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    # First on real words
    with timer("Tfidf on word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            max_features=20000)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)

    del word_vectorizer
    gc.collect()

    # Now use the char_analyzer to get another TFIDF
    # Char level TFIDF would go through words when char analyzer only considers
    # characters inside a word
    with timer("Tfidf on char n_gram"):
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)
        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

    del char_vectorizer
    gc.collect()

    print((train_char_features > 0).sum(axis=1).max())

    del train_text
    del test_text
    gc.collect()

    # Now stack TF IDF matrices
    with timer("Staking matrices"):
        csr_trn = hstack(
            [
                train_char_features,
                train_word_features,
                train_num_features
            ]
        ).tocsr()
        # del train_word_features
        del train_num_features
        del train_char_features
        gc.collect()

        csr_sub = hstack(
            [
                test_char_features,
                test_word_features,
                test_num_features
            ]
        ).tocsr()
        # del test_word_features
        del test_num_features
        del test_char_features
        gc.collect()
    submission = pd.DataFrame.from_dict({'id': test['id']})
    del test
    gc.collect()

    # Drop now useless columns in train and test
    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)
    gc.collect()

    # Set LGBM parameters
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.0875,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1
    }

    # Now go through folds
    # I use K-Fold for reasons described here : 
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    with timer("Scoring Light GBM"):
        scores = []
        folds = KFold(n_splits=10, shuffle=True, random_state=233)
        lgb_round_dict = defaultdict(int)
        trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
        del csr_trn
        gc.collect()
        
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)
            
            lgb_rounds = 1000

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                watchlist = [
                    trn_lgbset.subset(trn_idx),
                    trn_lgbset.subset(val_idx)
                ]
                # Train lgb l1
                model = lgb.train(
                    params=params,
                    train_set=watchlist[0],
                    num_boost_round=lgb_rounds,
                    valid_sets=watchlist,
                    early_stopping_rounds=30,
                    verbose_eval=0
                )
                class_pred[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])
                
                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                lgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
            
            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred

        # Save OOF predictions - may be interesting for stacking...
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv("lgbm_kernel_oof_preds.csv",
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

    with timer("Predicting probabilities"):
        # Go through all classes and reuse computed number of rounds for each class
        for class_name in class_names:
            with timer("Predicting probabilities for %s" % class_name):
                train_target = train[class_name]
                trn_lgbset.set_label(train_target.values)
                # Train lgb
                model = lgb.train(
                    params=params,
                    train_set=trn_lgbset,
                    num_boost_round=int(lgb_round_dict[class_name] / folds.n_splits)
                )
                submission[class_name] = model.predict(csr_sub, num_iteration=model.best_iteration)

submission.to_csv("lgbm_kernel_sub_preds.csv", index=False, float_format="%.8f")