
# coding: utf-8

# ## Instracart Market EDA - (a)

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# ### Data overview

# In[2]:

os.chdir('C:/Users/Bangda/Desktop/kaggle/instacart-market')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
products = pd.read_csv('products.csv')
orders = pd.read_csv('orders.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
order_products_train = pd.read_csv('order_products__train.csv')


# In[3]:

aisles.shape, departments.shape, products.shape, orders.shape, order_products_prior.shape, order_products_train.shape


# In[4]:

aisles.head()


# In[5]:

departments.head()


# In[6]:

products.head()


# In[7]:

orders.head() # order_dow is the day of a week


# In[8]:

order_products_prior.head()


# In[9]:

order_products_train.head()


# In[10]:

test = orders.loc[orders.eval_set == 'test']
train = orders.loc[orders.eval_set == 'train']
prior = orders.loc[orders.eval_set == 'prior']
test.shape, train.shape, prior.shape


# In[11]:

test.head()


# In[12]:

# reordered days
orders.days_since_prior_order.value_counts()


# In[13]:

orders.days_since_prior_order.isnull().sum()


# In[14]:

# order_id is the identifier, order_number is quantity
orders.describe()


# #### Append user_id to order_products_prior/table

# In[15]:

# number of different product purchased for each user
order_products_prior = pd.merge(order_products_prior, prior[['order_id', 'user_id']], 
                                on = 'order_id', how = 'left')
order_products_train = pd.merge(order_products_train, train[['order_id', 'user_id']],
                                on = 'order_id', how = 'left')


# In[16]:

order_products_prior.head()


# ### Interpret prior set, train set and test set

# In[17]:

# interpret prior, train, test: take user_id == 1 as example
train.loc[train.user_id == 1]


# In[18]:

prior.loc[prior.user_id == 1]


# In[19]:

test.head()


# In[20]:

# prior purchase record (product) for user_id 1
order_products_prior.loc[order_products_prior.user_id == 1].product_id.value_counts()


# In[21]:

# "next" purchase record of user_id 1. we need to predict this from the prior
order_products_train.loc[order_products_train.user_id == 1]


# In[22]:

# products purchased again by user_id 1
buy_prev = order_products_train.loc[order_products_train.user_id == 1].product_id.values
buy_next = order_products_prior.loc[order_products_prior.user_id == 1].product_id.values
print("user_id 1: prior purchased items:\n{}".format(buy_next))
print("user_id 1: item purchased again (in train):\n{}".format(np.intersect1d(buy_prev, buy_next)))


# ### Re-format the table as: user_id | prior_purchased_items

# In[23]:

# start from table: 
# in this table, we can view all distinct product purchased by each user_id
# .. user_id .. | .. product_id ..
# what we want is table like this: user_id | [product_id list]
# order_products_prior_distinct_userid = order_products_prior.groupby(['user_id'])['product_id'].apply(list)
# buy_prev_userid_1 = order_products_prior_distinct_userid[order_products_prior_distinct_userid.index == 1]
# buy_prev_userid_1.values


# In[24]:

# prior_userid_product_lst = pd.DataFrame(np.stack((order_products_prior_distinct_userid.index, 
#                                                   order_products_prior_distinct_userid), axis = 1),
#                                        columns = ['user_id', 'product_id_list_prior'])
# prior_userid_product_lst['product_id_list_prior_count'] = prior_userid_product_lst['product_id_list_prior'].apply(lambda x: len(x))
# prior_userid_product_lst.head()


# In[25]:

# order_products_train_distinct_userid = order_products_train.groupby(['user_id'])['product_id'].apply(list)
# train_userid_product_lst = pd.DataFrame(np.stack((order_products_train_distinct_userid.index, 
#                                                   order_products_train_distinct_userid), axis = 1),
#                                         columns = ['user_id', 'product_id_list_train'])
# train_userid_product_lst['product_id_list_train_count'] = train_userid_product_lst['product_id_list_train'].apply(lambda x: len(x))
# train_userid_product_lst.head()


# ### Reordered = 0 removed

# In[26]:

order_products_prior_reordered = order_products_prior.loc[order_products_prior.reordered != 0]
order_products_train_reordered = order_products_train.loc[order_products_train.reordered != 0]
order_products_prior.shape, order_products_prior_reordered.shape, order_products_train.shape, order_products_train_reordered.shape


# In[27]:

order_products_prior_distinct_userid_reordered = order_products_prior_reordered.groupby(['user_id'])['product_id'].apply(list)
prior_userid_product_reordered_lst = pd.DataFrame(np.stack((order_products_prior_distinct_userid_reordered.index, 
                                                            order_products_prior_distinct_userid_reordered), axis = 1),
                                                  columns = ['user_id', 'product_id_list_prior'])
prior_userid_product_reordered_lst['product_id_list_train_count'] = prior_userid_product_reordered_lst['product_id_list_prior'].apply(lambda x: len(x))
prior_userid_product_reordered_lst.head()


# In[28]:

order_products_train_distinct_userid_reordered = order_products_train_reordered.groupby(['user_id'])['product_id'].apply(list)
train_userid_product_reordered_lst = pd.DataFrame(np.stack((order_products_train_distinct_userid_reordered.index, 
                                                            order_products_train_distinct_userid_reordered), axis = 1),
                                                  columns = ['user_id', 'product_id_list_train'])
train_userid_product_reordered_lst['product_id_list_train_count'] = train_userid_product_reordered_lst['product_id_list_train'].apply(lambda x: len(x))
train_userid_product_reordered_lst.head()


# ### ...started at 9/7/2017

# In[29]:

# number of users: user numbers in train and test are consistent with size; user_id in prior is not distinct
prior['user_id'].unique().size, train['user_id'].unique().size, test['user_id'].unique().size


# In[30]:

order_products_train.shape, order_products_prior.shape


# In[31]:

# number of users in order_products_train and order_products_prior are consistent with in prior, train
order_products_train['user_id'].unique().size, order_products_prior['user_id'].unique().size 


# In[32]:

order_products_prior.head()


# In[33]:

order_products_prior.info()


# In[34]:

# remove reoredered == 0 in order_products_prior
order_products_prior_reordered = order_products_prior.loc[order_products_prior['reordered'] > 0]
order_products_prior_reordered.shape


# In[35]:

# try to format like this: user_id (distinct) | product_id (distinct) | count
user_product_count = order_products_prior_reordered.groupby(['user_id', 'product_id'])[['product_id']].count()
user_product_count.columns = ['product_count']
user_product_count.shape


# In[36]:

user_product_count = user_product_count.reset_index()
user_product_count.head()


# In[37]:

print('Number of users in prior (reordered): {}'.format(user_product_count.user_id.unique().shape[0]))
print('Number of products in prior (reordered): {}'.format(user_product_count.product_id.unique().shape[0]))


# In[38]:

# Try to form a matrix where each row represents one user and each column represents one product
# MEMORY ERROR
# user_product_matrix = np.zeros((user_product_count.user_id.unique().shape[0], user_product_count.product_id.unique().shape[0]))
# for row in user_product_count.itertuples():
#     user_product_matrix[row[1] - 1, row[2] - 1] = row[3]


# In[39]:

user_product_count_subset = user_product_count.iloc[:20000, :]
print('Number of users in prior subset (reordered): {}'.format(user_product_count_subset.user_id.unique().shape[0]))
print('Number of products in prior subset (reordered): {}'.format(user_product_count_subset.product_id.unique().shape[0]))


# In[40]:

user_product_subset_matrix = np.zeros((user_product_count_subset.user_id.unique().shape[0], user_product_count_subset.product_id.unique().shape[0]))
user_product_subset_matrix.shape
#for row in user_product_count_subset.itertuples():
#    user_product_subset_matrix[row[1] - 1, row[2] - 1] = row[3]


# In[41]:

user_product_raw_id = user_product_count_subset['product_id'].value_counts().index.values
user_product_raw_id = pd.DataFrame(user_product_raw_id)
user_product_raw_id['new_product_id'] = np.arange(7132) # user_product_subset_matrix.shape[1]
user_product_raw_id.columns = ['raw_product_id', 'new_product_id']
user_product_raw_id.head()

