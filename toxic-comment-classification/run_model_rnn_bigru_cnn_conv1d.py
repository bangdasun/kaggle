
# This is refer from public kernel:
# https://www.kaggle.com/konohayui/bi-gru-cnn-poolings
import os
import sys
import re
import gc
import csv
import logging
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings('ignore')

from toxic_utils import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing import text, sequence
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras import initializers, regularizers, constraints, callbacks, optimizers, layers, callbacks
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import PReLU, BatchNormalization
from keras.models import Model, load_model
from keras.engine import InputSpec, Layer
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU


# -------------------------------------
# --- utility functions and classes ---
# -------------------------------------
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')			


EMBEDDING_FILE = 'c:\\users\\bangda\\documents\\toxic-comment-classification\\crawl-300d-2M.vec\\crawl-300d-2M.vec'
embed_size   = 300
max_features = 150000
max_len      = 200

# -----------------
# --- read data ---
# -----------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# --------------------------
# --- text preprocessing ---
# --------------------------
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

train["comment_text"].fillna("no comment")
test["comment_text"].fillna("no comment")

train.comment_text = train.comment_text.str.lower()
test.comment_text = test.comment_text.str.lower()

train.comment_text = train.comment_text.apply(lambda x: clean_url(x))
test.comment_text = test.comment_text.apply(lambda x: clean_url(x))

train.comment_text = train.comment_text.apply(lambda x: clean_comment(x))
test.comment_text = test.comment_text.apply(lambda x: clean_comment(x))

train.comment_text = train.comment_text.apply(lambda x: character_range(x))
test.comment_text = test.comment_text.apply(lambda x: character_range(x))

str_replace(train)
str_replace(test)

X_train_all = train

raw_text_train = X_train_all["comment_text"]
raw_text_test = test["comment_text"]

# tokenization
tk = Tokenizer(num_words=max_features, lower=True)
tk.fit_on_texts(raw_text_train)
X_train_all.loc[:, "comment_seq"] = tk.texts_to_sequences(raw_text_train)
test.loc[:, "comment_seq"] = tk.texts_to_sequences(raw_text_test)

# padding to fixed length
X_train_all = pad_sequences(X_train_all.comment_seq, maxlen=max_len)
test = pad_sequences(test.comment_seq, maxlen=max_len)


# -------------------------------------
# --- apply word-embeddings on text ---
# -------------------------------------
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8', mode='r'))
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)

	
# --------------------
# --- define model ---
# --------------------
def build_model(X_train, Y_train, X_valid, Y_valid, lr=0.0, lr_d=0.0, units=0, dr=0.0):
    
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    # GRU could be replaced by LSTM
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    # could remove the Conv1D layer
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    
    for bs in [128, 256]:
        history = model.fit(X_train, Y_train, batch_size=bs, epochs=1, validation_data=(X_valid, Y_valid), 
                           verbose=1, callbacks=[ra_val])
    return model

	
# -----------------
# --- run model ---
# -----------------
n_folds = 10
folds = KFold(n_splits=n_folds, shuffle=True, random_state=233)
preds = []
oof_preds = np.zeros((X_train_all.shape[0], 6))

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train_all)):
    
    X_train, X_valid = X_train_all[trn_idx], X_train_all[val_idx]
    Y_train, Y_valid = y[trn_idx], y[val_idx]
	
    # define callbacks
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval=1)
    
    print('\n------ Fold {} ------\n'.format(n_fold + 1))
    model = build_model(X_train, Y_train, X_valid, Y_valid, lr=1e-3, lr_d=1.5e-5, units=128, dr=0.3)
    
    print('------ Make OOF predictions ------')
    oof_preds[val_idx, :] = model.predict(X_valid, batch_size=1024, verbose=1)
    
    print('------ Make predictions ------')
    preds.append(model.predict(test, batch_size=1024, verbose=1))
	
# --------------------------
# --- submit predictions ---
# --------------------------
for i, c in enumerate(list_classes):
    train[c + '_oof'] = oof_preds[:, i]
	
train[['id'] + list_classes + [f + '_oof' for f in list_classes]].to_csv('rnn_bigru_cnn_conv1d_oof_preds.csv', index=False)

pred = sum(preds) / n_folds
submission = pd.read_csv("sample_submission.csv")
submission[list_classes] = pred
submission.to_csv('rnn_bigru_cnn_conv1d_sub_preds.csv', index=False)
