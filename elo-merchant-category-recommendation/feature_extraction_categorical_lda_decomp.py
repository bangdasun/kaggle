
import pickle
import argparse
import pandas as pd
from kaggle_learn.utils import timer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('categorical_feature', help='categorical feature name')
    parser.add_argument('ngram', help='max ngram')
    parser.add_argument('max_features', help='maximum number of features')
    parser.add_argument('n_components', help='number of components')
    args = parser.parse_args()

    feat = str(args.categorical_feature)
    ngram = int(args.ngram)
    max_features = int(args.max_features)
    n_components = int(args.n_components)

    with timer('Load data'):
        hist = pd.read_csv('hist_transac_processed.csv')
        print(hist.shape)

    with timer('Convert {} to sequence'.format(feat)):
        hist[feat] = hist[feat].astype(str)
        hist_feat_seq = hist.sort_values('purchase_date').groupby('card_id')[feat].apply(list)
        hist_feat_seq = hist_feat_seq.reset_index()
        hist_feat_seq.columns = ['card_id', 'hist_{}_seq'.format(feat)]
        hist_feat_seq['hist_{}_seq'.format(feat)] = hist_feat_seq['hist_{}_seq'.format(feat)].apply(lambda x: ' '.join(x))

    with timer('Vectorizing {} sequence'.format(feat)):
        vectorizer = CountVectorizer(token_pattern='\w+', ngram_range=(1, ngram), max_features=max_features)
        hist_feat_seq_vec = vectorizer.fit_transform(hist_feat_seq['hist_{}_seq'.format(feat)].values)
        print(hist_feat_seq_vec.shape)

    with timer('Save {} vectors'.format(feat)):
        with open('hist_{}_seq_vec.pkl'.format(feat), 'wb') as f:
            pickle.dump(hist_feat_seq_vec, f)

    with timer('LDA decomposition'):
        lda = LDA(n_components=n_components, random_state=4590)
        hist_feat_lda_comp = lda.fit_transform(hist_feat_seq_vec)
        print(hist_feat_lda_comp.shape)

    with timer('Save decomposition features'):
        hist_feat_lda_comp_df = pd.DataFrame()
        hist_feat_lda_comp_df['card_id'] = hist_feat_seq['card_id']
        for i in range(n_components):
            hist_feat_lda_comp_df['hist_{}_lda_comp_{}'.format(feat, i + 1)] = hist_feat_lda_comp[:, i]
        print(hist_feat_lda_comp_df.shape)
        hist_feat_lda_comp_df.to_csv('hist_transac_{}_lda_comp_0_1.csv'.format(feat), index=False)