
# This is my private kernel fork from https://www.kaggle.com/christofhenkel/fasttext-starter-description-only
import os
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
from keras.preprocessing import text, sequence
from tqdm import tqdm
from keras.layers import Input, concatenate, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding, GlobalMaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
                    input_length = maxlen, trainable = False)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(128, return_sequences = True))(main)
    main = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(main)
    max_pool = GlobalMaxPooling1D()(main)
    avg_pool = GlobalAveragePooling1D()(main)
    main = concatenate([avg_pool, max_pool])
    out = Dense(1, activation = "relu")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    model.summary()
    return model

# Could use self-trained embeddings: https://www.kaggle.com/christofhenkel/self-trained-embeddings-starter-only-description
EMBEDDING_FILE = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
TRAIN_CSV = '../input/avito-demand-prediction/train.csv'
TEST_CSV = '../input/avito-demand-prediction/test.csv'

max_features = 100000
maxlen = 120
embed_size = 300

train = pd.read_csv(TRAIN_CSV, index_col = 0)
train['description'] = train['parent_category_name'].fillna('') + ' ' + \
                       train['category_name'].fillna('') + ' ' + \
                       train['description'].fillna('') + ' ' + \
                       train['param_1'].fillna('') + ' ' + \
                       train['param_2'].fillna('') + ' ' + \
                       train['param_3'].fillna('')

labels = train[['deal_probability']].copy()
train = train[['description']].copy()

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')

train['description'] = train['description'].astype(str)
tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))


print('getting embeddings')
def get_coefs(word, *arr): 
	return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del embeddings_index

folds = KFold(n_splits=5, shuffle=True, random_state=42)
EPOCHS = 3

test = pd.read_csv(TEST_CSV, index_col = 0)
test['description'] = test['parent_category_name'].fillna('') + ' ' + \
                       test['category_name'].fillna('') + ' ' + \
                       test['description'].fillna('') + ' ' + \
                       test['param_1'].fillna('') + ' ' + \
                       test['param_2'].fillna('') + ' ' + \
                       test['param_3'].fillna('')

test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
oof_preds = []
sample_submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv', index_col = 0)
submission = sample_submission.copy()


for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train['description'].values)):
    print('='*20, ' Fold', n_fold + 1, '='*20)
    X_train = train['description'].iloc[trn_idx].values
    y_train = labels['deal_probability'].iloc[trn_idx].values
    
    X_valid = train['description'].iloc[val_idx].values
    y_valid = labels['deal_probability'].iloc[val_idx].values
    
    
    print('convert to sequences')
    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)

    print('padding')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

    model = build_model()
    file_path = "model_{}.hdf5".format(n_fold + 1)

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 2)
    history = model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 2, callbacks = [check_point])
    model.load_weights(file_path)
    oof_prediction = model.predict(X_valid)
    oof_prediction = np.clip(oof_prediction, 0.0, 1.0)
    oof_preds_df = pd.DataFrame(val_idx, columns=['idx'])
    oof_preds_df['rnn_preds'] = oof_prediction
    oof_preds.append(oof_preds_df) 
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, oof_prediction)))


    prediction = model.predict(X_test,  batch_size = 128, verbose = 2)
    prediction = np.clip(prediction, 0.0, 1.0)

    submission.loc[:, 'deal_probability_{}'.format(n_fold+1)] = prediction


submission.to_csv('rnn_conv_sub_preds.csv')

with open('rnn_conv_oof_preds.pkl', 'wb') as f:
    pickle.dump(oof_preds, f)
