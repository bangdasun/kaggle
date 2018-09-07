
import os
import gc
import pickle
import time
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from kaggle_learn.utils import timer, memory_uage, reduce_memory_usage
from kaggle_learn.feature_engineering.statistics import *
from kaggle_learn.feature_engineering.text import * 


# ============================================
# Helper functions
# ============================================

def search_exclamation(x):
    result = re.search('!', x)
    if result:
        return result.start()
    else:
        return -1


def str_common_words(title, desc):
    words, cnt = title.split(), 0
    for w in words:
        if desc.find(w) >= 0:
            cnt += 1
    return cnt


def get_categorical_emb(df, cat1, cat2, n, method):
    cat2_of_cat1 = {}
    method_map = {'lda': LDA, 'nmf': NMF, 'svd': TruncatedSVD}

    for row in df.iterrows():
        cat2_of_cat1.setdefault(str(row[1][cat1]), []).append(str(row[1][cat2]))
    cat1 = list(cat2_of_cat1.keys())
    cat2_as_sent = [' '.join(cat2_of_cat1[c]) for c in cat1]
    vectorizer = CountVectorizer()
    cat2_as_matrix = vectorizer.fit_transform(cat2_as_sent)

    return method_map['method'](n_components=n).fit_transform(cat2_as_matrix), cat1


def merge_emb_df(df, emb_matrix, cat1, cat2, cat1_list, post_fix=''):
    emb_df = pd.DataFrame(emb_matrix)
    emb_df.columns = ['emb_' + cat1 + '_' + cat2 + '_' + str(i) + post_fix for i in range(1, emb_df.shape[1] + 1)]
    emb_df[cat1] = cat1_list
    emb_df[cat1] = emb_df[cat1].astype('int')
    df = df.merge(emb_df, on=cat1, how='left')
    return df


def add_emb_matrix(df, cat1, cat2, n, method, post_fix):
    # get embedding matrix could take long time, save it when finish and read it when use it
	filename = 'emb_{}_{}_features.csv'.format(cat1, cat2)
	if os.path.exists(filename):
		emb = pd.read_csv(filename)
		df = pd.concat([df, emb], axis=1)
		emb_features = ['emb_{}_{}_{}'.format(cat1, cat2, i) for i in range(1, n + 1)]
	else:
		user_cat_emb, cat1_list = get_categorical_emb(df, cat1, cat2, n, method)
		df = merge_emb_df(df, user_cat_emb, cat1, cat2, cat1_list, post_fix)
		emb_features = ['emb_{}_{}_{}'.format(cat1, cat2, i) for i in range(1, n + 1)]
		df[emb_features[-n:]].to_csv(filename, index=False)
	return df


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
	
	@author: Serg @address: https://www.kaggle.com/rumbok

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
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


# from https://www.kaggle.com/classtag/cat2vec-powerful-feature-for-categorical
def apply_w2v(sentences, model, num_features):

    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.
        for word in words:
            if word in vocabulary:
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])
        
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    vocab = set(model.wv.index2word)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return csr_matrix(np.array(feats))


def merge_target_cluster(df, category, target, ntrain):
    
    gp = df.iloc[:ntrain].loc[:, [category, target]].groupby(category)[target]
    hist = gp.agg(lambda x: ' '.join(str(x)))
    gp_index = hist.index
    
    sentences = [x.split(' ') for x in hist.values]
    n_features = 500
    w2v = Word2Vec(sentences=sentences, min_count=1, size=n_features)
    w2v_features = apply_w2v(sentences, w2v, n_features)
    cluster_labels = KMeans(n_clusters=20).fit(w2v_features).labels_
    cluster_labels = pd.Series(cluster_labels, name=category+'_cluster',
                              index=gp_index)
    return df[category].map(cluster_labels).fillna(-1).astype(int)


# ============================================
# Feature Extraction
# ============================================

with timer('Load data'):
	train = pd.read_csv('train.csv.zip', parse_dates = ["activation_date"])
	test = pd.read_csv('test.csv.zip', parse_dates = ["activation_date"])
	ntrain = train.shape[0]
	ntest = test.shape[0]
	train['num_na'] = train.isnull().sum(axis=1)
	test['num_na'] = test.isnull().sum(axis=1)
	df = pd.concat([train, test], axis=0)
	del train, test
	gc.collect()

with timer('Simple feature engineering'):
    df['dow'] = df['activation_date'].dt.weekday
    df['dom'] = df['activation_date'].dt.day
    df['param_1_len'] = df['param_1'].apply(lambda x: calc_len(x))
    
    # features from https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm
    gp = pd.read_csv('aggregated_features_1.csv')
    df = df.merge(gp, on='user_id', how='left')
    del gp; gc.collect()

with timer('Add target cluster features'):
    df['category_name_prob_cluster'] = merge_target_cluster(df, 'category_name', 'deal_probability', ntrain)
    df['image_top_1_prob_cluster'] = merge_target_cluster(df, 'image_top_1', 'deal_probability', ntrain)
    df['param_1_prob_cluster'] = merge_target_cluster(df, 'param_1', 'deal_probability', ntrain)
    df['param_2_prob_cluster'] = merge_target_cluster(df, 'param_2', 'deal_probability', ntrain)
    df['category_name_price_cluster'] = merge_target_cluster(df, 'category_name', 'price', ntrain)
    df['image_top_1_price_cluster'] = merge_target_cluster(df, 'image_top_1', 'price', ntrain)
    df['param_1_price_cluster'] = merge_target_cluster(df, 'param_1', 'price', ntrain)
    df['param_2_price_cluster'] = merge_target_cluster(df, 'param_2', 'price', ntrain)


with timer('Append category to desc / param to title'):
    df['description'] = df['description'].fillna('') + ' ' + \
                        df['parent_category_name'].fillna('') + ' ' + \
                        df['category_name'].fillna('')
            
    df['title'] = df['title'].fillna('') + ' ' + \
                  df['param_1'].fillna('') + ' ' + \
                  df['param_2'].fillna('') + ' ' + \
                  df['param_3'].fillna('')


with timer('Encoding categorical features'):
	categorical_features = ['user_id', 'region', 'city', 'parent_category_name', 'category_name',
							'user_type', 'image_top_1', 'param_1', 'param_2', 'param_3']
	text_features = ['title', 'description']
	lbl = LabelEncoder()
    for c in categorical_features:
        df[c] = lbl.fit_transform(df[c].astype(str))

with timer('Update categorical features'):
    categorical_features.extend(['category_name_prob_cluster',
                                 'image_top_1_prob_cluster',
                                 'param_1_prob_cluster', 
                                 'param_2_prob_cluster',
                                 'category_name_price_cluster',
                                 'image_top_1_price_cluster',
                                 'param_1_price_cluster', 
                                 'param_2_price_cluster'])

with timer('Extracting text-statistical features'):
    df['description'] = df['description'].fillna('nan')
    df['description_num_chars'] = df['description'].apply(len)
    df['description_num_words'] = df['description'].apply(lambda x: len(x.split()))
    df['description_num_unique_words'] = df['description'].apply(lambda x: len(set(x.split())))
    df['description_num_sent'] = df['description'].apply(lambda x: len(re.split('\\.{1,}|\\!{1,}|\n{1,}', x)))
    df['description_num_exclamation'] = df['description'].apply(lambda x: x.count('!') ** 2)
    df['description_num_capitals'] = df['description'].apply(lambda x: np.sum(1 for c in x if c.isupper()))
    df['description_words_avg_len'] = df['description_num_chars'] / df['description_num_words']
    df['description_unique_words_ratio'] = df['description_num_unique_words'] / df['description_num_words']
    df['description_sen_avg_len'] = df['description_num_words'] / df['description_num_sent']
    df['description_num_punct'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['description_punct_words_ratio'] = df['description_num_punct'] / df['description_num_chars']
    df['description_pos_exclamation'] = df['description'].apply(lambda x: search_exclamation(x)) / df['description_num_chars']
    df['description_num_digit'] = df['description'].apply(lambda x: np.sum(1 for c in x if c.isdigit()))
    df['description_num_digit_ratio'] = df['description_num_digit'] / (df['description_num_chars'] + 1)
    
    df['title'] = df['title'].fillna('nan')
    df['title_num_chars'] = df['title'].apply(len)
    df['title_num_words'] = df['title'].apply(lambda x: len(x.split()))
    df['title_num_unique_words'] = df['title'].apply(lambda x: len(set(x.split())))
    df['title_words_avg_len'] = df['title_num_chars'] / df['title_num_words']
    df['title_description_len_ratio'] = df['title_num_chars'] / (df['description_num_chars'] + 1)
    
    df['num_title_in_desc'] = df[['title', 'description']].apply(lambda x: str_common_words(x['title'], x['description']), axis=1)
    df['num_title_in_desc_ratio'] = df['num_title_in_desc'] / df['title_num_words']
    df['param_1_title_len_ratio'] = df['param_1_len'] / df['title_num_chars']


with timer('Groupby simple statistical features'):
    agg_func_map = {
    	'mean'    : add_group_mean,
    	'count'   : add_group_count,
    	'nunique' : add_group_nunique,
    	'median'  : add_group_median,
    	'std'     : add_group_std,
    	'cumcount': add_group_cumcount,
    	'entropy' : add_group_entropy
    }

    agg_config = [
    	(['region', 'city'], [('item', 'count'),
			      ('user_id', 'nunique')]),
    	(['region', 'city', 'activation_date'], [('item_id', 'count')]),
    	(['user_id'], [('title', 'nunique')]),
    	(['user_id', 'param_1'], [('item_id', 'count')]),
    	(['user_id', 'activation_date'], [('item_id', 'count')]),
		(['category_name'], [('item_id', 'nunique')]),
    	(['category_name', 'param_1'], [('user_id', 'count'),
					('user_id', 'nunique')]),
    	(['category_name', 'image_top_1'], [('user_id', 'count'),
					    ('user_id', 'nunique')]),
    	(['category_name', 'city'], [('user_id', 'count'),
				     ('user_id', 'nunique'),
				     ('item_id', 'nunique')])
    	(['category_name', 'city', 'activation_date'], [('item_id', 'nunique')]),
    	(['image_top_1', 'param_1'], [('user_id', 'count'),
				      ('user_id', 'nunique')]),
    	(['image_top_1', 'city'], [('user_id', 'count'),
				   ('user_id', 'nunique')]),
    	(['parent_category_name', 'city'], [('category_name', 'count')])
    ]

    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            df = agg_func_map[agg_feat[1]](df, cols=agg_pair[0], cname=new_col, value=agg_feat[0])

    df = merge_cumcount(df, cols=['user_id', 'activation_date'], cname='user_item_date_cumcount')

    # add_group_entropy(df, group, subgroup, cname, value)
    df = merge_entropy(df, 'user_id', 'parent_category_name', 'user_id_parent_category_name_entropy', 'category_name')
    df = merge_entropy(df, 'user_id', 'category_name', 'user_id_category_name_entropy', 'parent_category_name')
    df = merge_entropy(df, 'user_id', 'image_top_1', 'user_id_image_top_1_entropy', 'category_name')
    df = merge_entropy(df, 'user_id', 'param_1', 'user_id_param_1_entropy', 'category_name')
    df = merge_entropy(df, 'user_id', 'activation_date', 'user_id_activation_date_entropy', 'category_name')
    df = merge_entropy(df, 'user_id', 'dow', 'user_id_dow_entropy', 'category_name')
    df = merge_entropy(df, 'city', 'dow', 'city_dow_entropy', 'category_name')
    df = merge_entropy(df, 'region', 'dow', 'region_dow_entropy', 'category_name')
    df = merge_entropy(df, 'category_name', 'city', 'category_city_entropy', 'item_id')
    df = merge_entropy(df, 'category_name', 'param_1', 'category_param_1_entropy', 'item_id')
    df = merge_entropy(df, 'category_name', 'image_top_1', 'category_image_top_1_entropy', 'item_id')
    df = merge_entropy(df, 'image_top_1', 'city', 'image_top_1_city_entropy', 'item_id')
    df = merge_entropy(df, 'image_top_1', 'param_1', 'image_top_1_param_1_entropy', 'item_id')
    df = merge_entropy(df, 'param_1', 'city', 'param_1_city_entropy', 'item_id')

    df['user_param_1_count_ratio'] = df['user_param_1_count'] / df['user_item_unique']

with timer('Categorical embedding features'):

	cat1_list = ['user_id', 'user_id', 'user_id', 'user_id', 'user_id', 'user_id', 'image_top_1', 'image_top_1']
	cat2_list = ['category_name', 'param_1', 'image_top_1', 'city', 'param_2', 'param_3', 'param_1', 'category_name']

	for cat1, cat2 in zip(cat1_list, cat2_list):
		df = add_emb_matrix(df, cat1, cat2, n=5, methpd='lda', post_fix='lda')
		df = add_emb_matrix(df, cat1, cat2, n=5, method='nmf', post_fix='nmf')

with timer('Reduce df to df_reduced'):
    df_reduced = reduce_mem_usage(df.iloc[:, 1:])
    del df; gc.collect()

with timer('Process title and description'):
    if os.path.exists('tfidf.pkl') and os.path.exists('vocab.pkl'):
        with open('tfidf.pkl', 'rb') as f:
            df_text_processed = pickle.load(f)
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
    	# Russian stopwords
	    russian_stop = set(stopwords.words('russian'))

	    tfidf_param = {
	        'stop_words'   : russian_stop,
	        'analyzer'     : 'word',
	        'token_pattern': r'\w{1,}',
	        'sublinear_tf' : True,
	        'dtype'        : np.float32,
	        'norm'         : 'l2',
	        'smooth_idf'   : False
	    }
	    
	    desc_features = 300000
	    title_features = 100000
	    
	    vectorizer = FeatureUnion([
	        ('description', TfidfVectorizer(
	            ngram_range=(1, 3),
	            max_features=desc_features,
	            min_df=3,
	            **tfidf_param,
	            preprocessor=get_col('description'))),
	        ('title', CountVectorizer(
	            ngram_range=(1, 2), 
	            stop_words=russian_stop,
	            max_features=title_features,
	            min_df=3,
	            preprocessor=get_col('title')))
	    ])

	    vectorizer.fit(df_reduced.to_dict('records'))
	    df_text_processed = vectorizer.transform(df_reduced.to_dict('records'))
	    vocab = vectorizer.get_feature_names()

with timer('Lda and nmf on title / description'):    
    if os.path.exists('desc_top_50_nmf.pkl'):
        desc_top_50 = np.load('desc_top_50_nmf.pkl')
        desc_top_50 = desc_top_50[:, :50]
        print('desc nmf features loaded')
    else:
        print('fit desc nmf features . . . ')
        desc_top_50 = NMF(n_components=50, max_iter=1000).fit_transform(df_text_processed[:, :desc_features])
        desc_top_50.dump('desc_top_50_nmf.pkl')
        
    if os.path.exists('title_top_50_lda.pkl'):
        title_top_50 = np.load('title_top_50_lda.pkl')
        print('title lda features loaded')
    else:
        print('fit title lda features . . .')
        title_top_50 = LDA(n_components=50, learning_method='online').fit_transform(df_text_processed[:, desc_features:])
        title_top_50.dump('title_top_50_lda.pkl')

with timer('Add title lda and desc nmf to data'):
    desc_nmf_features = ['desc_nmf_' + str(i) for i in range(1, 51)]
    title_lda_features = ['title_lda_' + str(i) for i in range(1, 51)]
    
    df_reduced_columns_before = df_reduced.columns.tolist()
    df_reduced = pd.concat([df_reduced, pd.DataFrame(desc_top_50, columns=[desc_nmf_features])], axis=1)
    df_reduced = pd.concat([df_reduced, pd.DataFrame(title_top_50, columns=[title_lda_features])], axis=1)
    df_reduced.columns = df_reduced_columns_before + desc_nmf_features + title_lda_features


with timer('Groupby price features'):
	df_reduced['price'] = df_reduced['price'].apply(lambda x: np.log1p(x))
    
    agg_config = [
    	(['region', 'city', 'dow'], [('price', 'mean'),
				     ('price', 'median'),
				     ('price', 'max'),
				     ('price', 'min'),
				     ('price', 'std')]),
    	(['user_id', 'param_1', 'dow'], [('price', 'mean'),
					 ('price', 'median'),
					 ('price', 'max'),
					 ('price', 'min'),
					 ('price', 'std')]),
    	(['user_id', 'image_top_1', 'dow'], [('price', 'mean'),
					     ('price', 'median'),
					     ('price', 'max'),
					     ('price', 'min'),
					     ('price', 'std')]),
    	(['user_id', 'category_name', 'dow'], [('price', 'mean'),
					       ('price', 'median'),
					       ('price', 'max'),
					       ('price', 'min'),
					       ('price', 'std')]),
    	(['user_id', 'region', 'city', 'category_name', 'dow'], [('price', 'mean'),
								 ('price', 'median'),
								 ('price', 'max'),
								 ('price', 'min'),
								 ('price', 'std')]),
    	(['region', 'city', 'parent_category_name', 'category_name'], [('price', 'mean'),
								       ('price', 'median'),
								       ('price', 'max'),
								       ('price', 'min'),
								       ('price', 'std')]),
    	(['category_name', 'image_top_1'], [('price', 'mean'),
					    ('price', 'max'),
					    ('price', 'min')]),
    	(['category_name', 'city'], [('price', 'mean'),
				     ('price', 'max'),
				     ('price', 'min')]),
    	(['category_name', 'image_top_1', 'city'], [('price', 'mean'),
						    ('price', 'max'),
						    ('price', 'min')]),
    	(['category_name', 'region'], [('price', 'mean'),
				       ('price', 'max')]),
    	(['category_name', 'image_top_1', 'regionon'], [('price', 'mean'),
							('price', 'max'),
							('price', 'min')]),
    	(['parent_category_name', 'image_top_1'], [('price', 'mean'),
						   ('price', 'max'),
						   ('price', 'min')]),
    	(['parent_category_name', 'city'], [('price', 'mean'),
					    ('price', 'max'),
					    ('price', 'min')]),
    	(['parent_category_name', 'image_top_1', 'city'], [('price', 'mean'),
							   ('price', 'max'),
							   ('price', 'min')]),
    	(['parent_category_name', 'region'], [('price', 'mean'),
					      ('price', 'max')]),
    	(['parent_category_name', 'image_top_1', 'regionon'], [('price', 'mean'),
							       ('price', 'max'),
							       ('price', 'min')])		
    ]

    new_cols = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_cols.append(new_col)
            df_reduced = agg_func_map[agg_feat[1]](df_reduced, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
            if agg_pair[0] == ['region', 'city', 'parent_category_name', 'category_name']:
            	df_reduced[new_col + '_diff'] = df_reduced[new_col] - df_reduced['price']


with timer('Advanced text features'):
	# Title desc similarity features
    df_reduced['title_desc_cos_dist'] = paired_cosine_distances(df_reduced[title_lda_features], df_reduced[desc_nmf_features])
    df_reduced['title_desc_mah_dist'] = paired_manhattan_distances(df_reduced[title_lda_features], df_reduced[desc_nmf_features])
    df_reduced['title_desc_eud_dist'] = paired_euclidean_distances(df_reduced[title_lda_features], df_reduced[desc_nmf_features])

    # Edit distance
    if os.path.exists('title_desc_edit_dist.csv'):
        df_reduced['title_desc_edit_dist'] = pd.read_csv('title_desc_edit_dist.csv', header=None).iloc[:, 0].values
    else:
        df_reduced['title_desc_edit_dist'] = df_reduced[['title', 'description']].apply(lambda x: dameraulevenshtein(x['title'], x['description']), axis=1)
        df_reduced['title_desc_edit_dist'].to_csv('title_desc_edit_dist.csv', index=False)

with timer('Feature selection'):
	try:
    	categorical_features.remove('user_id')
	except:
    	print('user_id already removed')
    
    # NEED TO BE IMPROVED
	features = df_reduced.columns.tolist()
	features.remove('deal_probability')
	features.remove('item_id')
	
	features.remove('emb_user_id_param_3_2_nmf')
	features.remove('emb_user_id_param_3_3_nmf')
	features.remove('desc_nmf_1')

	target = 'deal_probability'
	print('number of features =', len(features))

with open('vocab.pkl', 'wb') as f:
	pickle.dump(vocab, f)

with open('tfidf.pkl', 'wb') as f:
	pickle.dump(df_text_processed, f)

with open('features.pkl', 'wb') as f:
	pickle.dump(features, f)

with open('categorical_features.pkl', 'wb') as f:
	pickle.dump(categorical_features, f)

with open('df_reduced.pkl', 'wb') as f:
	pickle.dump(df_reduced, f)
