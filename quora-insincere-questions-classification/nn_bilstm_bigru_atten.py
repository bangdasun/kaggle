"""
    The pytorch based code fork from:
    @bminixhofer
    https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch

"""

import os
import gc
import time
import string
import random
import logging
import regex as re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data


# ------------------------
# --- global variables ---
# ------------------------
embed_size = 300
seed = 2018
embedding_file_dct = {
    'glove'   : '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
    'wiki'    : '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
    'paragram': '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
}

puncts = ['☹', '＞', 'ξ', 'ட', '「',  '½', '△', 'É', '¿', 'ł',
		'¼', '∆', '≥', '⇒', '¬', '∨', 'č', 'š', '∫', '▾', 'Ω', 
		'＾', 'ý', 'µ', '?', '!', '.', ',', '"', '#', '$', '%',
		'\\', "'", '(', ')', '*', '+', '-', '/', ':', ';', '<',
		'=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', 
		'~', '“', '”', '’', '′', '…', 'ɾ', '̃', 'ɖ', '–', '‘',
		'√', '→',  '—', '£', 'ø', '´', '×', 'í', '÷', 'ʿ', '€',
		'ñ', 'ç', 'へ', '↑', '∞', 'ʻ', '℅''ι', '•', 'ì', '−', '∈',
		'∩', '⊆', '≠', '∂', 'आ', 'ह', 'भ', 'ी', '³', 'च', '...', 
		'⌚', '⟨', '⟩', '∖', '˂',  '☺', 'ℇ', '❤', '♨', '✌', 'ﬁ', 
		'て', '„', '¸', 'ч',  '⧼', '⧽', 'ম', 'হ', 'ῥ', 'ζ', 'ὤ',
		'Ü', 'Δ',  'ʃ', 'ɸ', 'ợ', 'ĺ', 'º', 'ष', '♭', '़', '✅', 
		'✓', '∘', '¨', '″', 'İ', '⃗', '̂', 'æ', 'ɔ', '∑', '¾', '≅',
		'‑', 'ֿ','ő', '－', 'ș', 'ן', 'Γ', '∪', '⊨', '∠', 'Ó', '«', 
		'»', 'Í', 'க', 'வ', 'ா', 'ம', '≈','،', '＝', '（', '）',
		'ə', 'ਨ', 'ਾ', 'ਮ', 'ੁ', '︠', '︡', 'ː', '∧', '∀', 'Ō', 'ㅜ',
		'™', 'ण', '≡',  '《', '》', 'ٌ', 'Ä', '」', ',', '.', '"', ':', 
		')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
		'>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·',
		'_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™',
		'›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
		'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░',
		'¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—',
		'‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀',
		'¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
		'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，',
		'♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
		'Ø', '¹', '≤', '‡', '√']
          
rm_puncts = ['µ', 'º', 'Ä', 'Í', 'Ó', 'Ü', 'ì', 'ý', 'İ', 'ĺ', 'Ō', 'ő', 
		'š', 'ɔ', 'ɖ', 'ə', 'ɸ', 'ɾ', 'ː',  'ण', '़','ম', 'হ',  'ਨ', 'ਮ', 
		'ੁ', 'க', 'ட', 'ம',  'வ',  'ா', 'ợ' ,'♭', '✓',]
          
for s in rm_puncts:
    puncts.remove(s)

puncts = list(set(puncts.copy()))

# -------------------------
# --- utility functions ---
# -------------------------
def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{}.log'.format(logger_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def set_seed(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def threshold_search(y_true, y_proba):
    """@author: ryanzhang @address: https://www.kaggle.com/ryanzhang
    """
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(10, 60)]:
        y_pred = y_proba > threshold
        score = f1_score(y_true=y_true, y_pred=y_pred)
        if threshold <= 0.4 and threshold >= 0.25:
            print('threshold = {}\tscore = {:.6f}, mean = {:.6f}'.format(threshold, score, np.mean(y_pred)))
            print('confusion matrix:\n{}\n'.format(confusion_matrix(y_true, y_pred)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
    
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    
def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    for s in puncts:
        text = text.replace(s, f' {s} ')
    return text
    
def clean_latex_tag(text):
    """
    https://www.kaggle.com/sunnymarkliu/latex-cannot-be-used-in-a-quora-question/comments
    """
    corr_t = []
    for t in text.split(" "):
        t = t.strip()
        if t != '':
            corr_t.append(t)
    text = ' '.join(corr_t)
    
    text = re.sub('(\[ math \]).+(\[ / math \])', 'mathematical formula', text)
    return text    

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def load_glove(word_index, max_features):
    embedding_file = embedding_file_dct['glove']
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(embedding_file))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index))
    print('Number of word_index = {}'.format(len(word_index)))
    print('Number of words = {}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
def load_para(word_index, max_features):
    embedding_file = embedding_file_dct['paragram']
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if len(o) > 100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index))
    print('Number of word_index = {}'.format(len(word_index)))
    print('Number of words = {}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
   
def preprocessing(X_train, X_test, params):
    max_len, max_features = params['max_len'], params['max_features']
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tokenizer.word_index
    set_seed()
    embedding_matrix_glove = load_glove(word_index, max_features=max_features)
    embedding_matrix_para = load_para(word_index, max_features=max_features)
    embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_para], axis=0)
    return X_train, X_test, embedding_matrix


# --------------
# --- models ---
# --------------
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class BiLstmGruAtten(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, emb_dropout=0.1,
                 max_len=70, max_features=95000, embed_size=300):
        super(BiLstmGruAtten, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dropout = emb_dropout
        self.max_len = max_len
        self.max_features = max_features
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(self.max_features, self.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = nn.Dropout2d(self.emb_dropout)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(self.hidden_size * 2, self.max_len)
        self.gru_attention = Attention(self.hidden_size * 2, self.max_len)
        
        self.linear_1 = nn.Linear(self.hidden_size * 8, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)
    
    def forward(self, X):
        h_embedding = self.embedding(X)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear_1(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
   
# ------------
# --- main ---
# ------------
def main(train, test, logger):
    
    logger.info('Cleaning text . . .')
    for df in [train, test]:
        df['question_text'] = df['question_text'].str.lower()
        df['question_text'] = df['question_text'].apply(clean_text)
        # df['question_text'] = df['question_text'].apply(clean_latex_tag)
    X_train_text = train['question_text'].fillna('_##_').values
    X_test_text = test['question_text'].fillna('_##_').values
    y_train = train['target'].values
    logger.info('Cleaning text complete')
    
    logger.info('Preprocessing for model . . .')
    params = {'max_len': 70, 'max_features': 200000}
    max_len = params['max_len']
    max_features = params['max_features']
    X_train_emb, X_test_emb, embedding_matrix = preprocessing(X_train_text, X_test_text, params)
    logger.info('embedding matrix shape: {}'.format(np.shape(embedding_matrix)))
    logger.info('Preprocessing complete')
    
    batch_size = 256
    epochs = 4
    
    do_cv = True
    if do_cv:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train_emb, y_train))
        train_preds = np.zeros((len(train)))
        test_preds = np.zeros((len(test)))
        set_seed()
        X_test_cuda = torch.tensor(X_test_emb, dtype=torch.long).cuda()
        X_test_tensor = torch.utils.data.TensorDataset(X_test_cuda)
        test_loader = torch.utils.data.DataLoader(X_test_tensor, batch_size=batch_size, shuffle=False)
        for idx, (train_idx, valid_idx) in enumerate(splits):
            logger.info('Fold {}'.format(idx + 1))
            X_train_fold = torch.tensor(X_train_emb[train_idx], dtype=torch.long).cuda()
            y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
            X_valid_fold = torch.tensor(X_train_emb[valid_idx], dtype=torch.long).cuda()
            y_valid_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
            
            logger.info('Build model')
            model = BiLstmGruAtten(embedding_matrix=embedding_matrix, hidden_size=96, emb_dropout=0.1, max_len=max_len, max_features=max_features, embed_size=300)
            model.cuda()
            
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters())
            
            train_tensor = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
            valid_tensor = torch.utils.data.TensorDataset(X_valid_fold, y_valid_fold)
            train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=False)
            
            logger.info('Training . . .')
            for epoch in range(epochs):
                start_time = time.time()
                model.train()
                avg_loss = 0.
                for x_batch, y_batch in train_loader:
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)
                
                model.eval()
                valid_preds_fold = np.zeros((X_valid_fold.size(0)))
                test_preds_fold = np.zeros((len(test)))
                avg_val_loss = 0.
                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    y_pred = model(x_batch).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
                elapsed_time = time.time() - start_time
                msg = f'Epoch {epoch + 1}/{epochs} train_loss = {avg_loss:.5f} val_loss = {avg_val_loss:.5f} time = {elapsed_time:.2f}s'
                print(msg)
                logger.info(msg)
                
            logger.info('Generating test predictions . . .')
            for i, (x_batch, ) in enumerate(test_loader):
                y_pred = model(x_batch).detach()
                test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            train_preds[valid_idx] = valid_preds_fold
            test_preds += test_preds_fold / len(splits)
                
        search_result = threshold_search(y_train, train_preds)
        msg = f'search result: {search_result}'
        print(msg)
        logger.info(msg)
    else:
        print('Simple train_test_split not implemented.')
    
    sub = test[['qid']].copy()
    print('Test predictions:')
    for p in np.arange(0.25, 0.4, 0.01):
        print('threshold = {:.2f}, prediction mean = {:.6f}'.format(p, np.mean(test_preds > p)))
    sub['prediction'] = test_preds > search_result['threshold']
    sub['prediction'] = test_preds > search_result['threshold']
    return sub
    
    
# -----------
# --- run ---
# -----------
if __name__ == '__main__':
    logger_main = get_logger('quora-main')
    logger_main.info('Start . . .')
    set_seed()
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    sub = main(train, test, logger_main)
    sub.to_csv('submission.csv', index=False)