
# coding: utf-8

# ## Instracart Market - EDA (b)

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

test = orders.loc[orders.eval_set == 'test']
train = orders.loc[orders.eval_set == 'train']
prior = orders.loc[orders.eval_set == 'prior']
test.shape, train.shape, prior.shape


# ### Check the number of items in each order_id for each user_id, etc

# In[4]:

# number of different product purchased for each user
order_products_prior = pd.merge(order_products_prior, prior[['order_id', 'user_id']], 
                                on = 'order_id', how = 'left')
order_products_train = pd.merge(order_products_train, train[['order_id', 'user_id']],
                                on = 'order_id', how = 'left')


# In[5]:

# user_id | .. order_id .. | .. order_id count ..
order_products_prior.groupby(['user_id', 'order_id'])[['order_id']].count()


# In[6]:

# number of items purchased of each user_id (not distinct)
# user_id | .. product count ..
order_products_prior.groupby(['user_id'])[['product_id']].count()


# In[7]:

# number of each items purchased by each user_id
# user_id | .. product_id .. | .. product count ..
prior_user_product_count = order_products_prior.groupby(['user_id'])['product_id'].value_counts()


# In[8]:

prior_user_product_count_df = pd.DataFrame(prior_user_product_count)
prior_user_product_count_df.head()

