
# This is refer from public kernel:
# https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras/data
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
from keras import initializers, regularizers, constraints, callbacks, optimizers, layers, callbacks
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, add
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import PReLU, BatchNormalization
from keras.models import Model, load_model
from keras.engine import InputSpec, Layer
from keras.models import Model
from keras.callbacks import Callback

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

# learing rate util function 
def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind] 
	

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
filter_nr        = 128
filter_size      = [2, 3, 5, 6]
max_pool_size    = 3
max_pool_strides = 2
dense_nr         = 512
spatial_dropout  = 0.35
dense_dropout    = 0.5
train_embed      = False

def build_model(X_train, Y_train, X_valid, Y_valid):

    comment = Input(shape=(max_len,))
    emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)
	
	# block 1
    block1 = Conv1D(filter_nr, kernel_size=filter_size[0], padding='same', activation='linear')(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size[0], padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment)
    resize_emb = PReLU()(resize_emb)
    
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)
    
	# block 2
    block2 = Conv1D(filter_nr, kernel_size=filter_size[1], padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size[1], padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    # block 3
    block3 = Conv1D(filter_nr, kernel_size=filter_size[2], padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size[2], padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
    
	# block 4
    block4 = Conv1D(filter_nr, kernel_size=filter_size[3], padding='same', activation='linear')(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size[3], padding='same', activation='linear')(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    output = add([block4, block3_output])
    output = GlobalMaxPooling1D()(output)
    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(6, activation='sigmoid')(output)
	
    model = Model(comment, output)
	model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(decay=1.5e-5), metrics=['accuracy'])
    
	lr = callbacks.LearningRateScheduler(schedule)
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
    
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid), 
              callbacks = [lr, ra_val] ,verbose=1)
	return model
	

# -----------------
# --- run model ---
# -----------------
batch_size = 128
epochs     = 4
n_folds    = 10
folds = KFold(n_splits=n_folds, shuffle=True, random_state=233)
preds = []
submission = pd.read_csv("sample_submission.csv")
oof_preds = np.zeros((X_train_all.shape[0], 6))

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train_all)):
    gc.collect()
    X_train, X_valid = X_train_all[trn_idx], X_train_all[val_idx]
    Y_train, Y_valid = y[trn_idx], y[val_idx]
    
    print('\n------ Fold {} ------\n'.format(n_fold + 1))
    model = build_model(X_train, Y_train, X_valid, Y_valid)
	
    print('------ Make OOF predictions ------')
    oof_preds[val_idx, :] = model.predict(X_valid, batch_size=1024, verbose=1)
    
    print('------ Make predictions ------')
    preds.append(model.predict(test, batch_size=1024, verbose=1))
	
	
# --------------------------
# --- submit predictions ---
# --------------------------
for i, c in enumerate(list_classes):
    train[c + '_oof'] = oof_preds[:, i]
	
train[['id'] + list_classes + [f + '_oof' for f in list_classes]].to_csv('dpcnn_conv1d_oof_preds.csv', index=False)

pred = sum(preds) / n_folds
submission = pd.read_csv("sample_submission.csv")
submission[list_classes] = pred
submission.to_csv('dpcnn_conv1d_sub_preds.csv', index=False)
