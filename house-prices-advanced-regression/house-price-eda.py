
# coding: utf-8

#### House Price Prediction - EDA and Feature Enigneering

# Load packages and datasets

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import mode, skew
get_ipython().magic(u'matplotlib inline')


# In[2]:

os.chdir('C:/Users/Bangda/Desktop/kaggle/housing-price0806')
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head(3)


# In[3]:

train.shape, test.shape


# In[4]:

test['SalePrice'] = -1


# Check variables

# In[5]:

train.info()


# In[6]:

# numerical variables
numerical_var = train.select_dtypes(exclude = [object]).columns
numerical_var


# In[7]:

# categorical variables
categorical_var = train.select_dtypes(include = [object]).columns
categorical_var


# In[8]:

# check missing data
all_data = [train, test]
na_count = []
for df in all_data:
    na_count.append(df.apply(lambda x: sum(x.isnull())))

for idx, df in enumerate(na_count):
    print('========================')
    print(na_count[idx][na_count[idx] > 0])


# In[9]:

# fill Alley, FireplaceQu, MiscFeature, Fence, PoolQC with 'No'
for df in all_data:
    df['Alley'].fillna('No', inplace = True)
    df['FireplaceQu'].fillna('No', inplace = True)
    df['MiscFeature'].fillna('No', inplace = True)
    df['Fence'].fillna('No', inplace = True)
    df['PoolQC'].fillna('No', inplace = True)

train['MiscFeature'].value_counts()


# In[10]:

# check missing data for each row
na_count_row = []
for df in all_data:
    na_count_row.append(df.apply(lambda x: sum(x.isnull()), axis = 1))

for idx, df in enumerate(na_count_row):
    print('========================')
    print(na_count_row[idx].sort_values(ascending = False)[:10])


# In[11]:

# select numerical variables only
train.select_dtypes(exclude = [object]).median()


# In[12]:

# fill missing data
fill_var_no = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
                        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType',
                        'MSSubClass', ]
fill_var_zero   = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']
fill_var_mode   = ['SaleType', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical', 'MSZoning']
fill_var_median = ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']

for df in all_data:
    df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    df['Functional'].fillna('Typical', inplace = True)
    df.drop(['Utilities'], axis = 'columns', inplace = True)
    
    for col in fill_var_no:
        df[col].fillna('No', inplace = True)
    
    for col in fill_var_zero:
        df[col].fillna(0, inplace = True)
    
    for col in fill_var_median:
        df[col].fillna(df[col].median(), inplace = True)
    
    for col in fill_var_mode:
        df[col].fillna(df[col].mode()[0], inplace = True)


# In[13]:

# check missing data again, should be none
na_count = []
for df in all_data:
    na_count.append(df.apply(lambda x: sum(x.isnull())))

for idx, df in enumerate(na_count):
    print('========================')
    print(na_count[idx][na_count[idx] > 0])


# In[14]:

# compare distribution of y, log(y)
mpl.rcParams['figure.figsize'] = 12, 9
plt.subplot(2, 1, 1)
sns.distplot(train['SalePrice'])
plt.subplot(2, 1, 2)
sns.distplot(np.log(train['SalePrice']))
plt.show()


# In[15]:

# calculate skewness
train['logSalePrice'] = np.log(train['SalePrice'])
print('Skewness of raw: {}'.format(skew(train['SalePrice'])))
print('Skewness of log-transformed: {}'.format(skew(train['logSalePrice'])))


# In[16]:

# take logarithm + create binary variable
for df in all_data:
    df['logLotArea']     = np.log(df['LotArea'] + 1)
    df['logTotalBsmtSf'] = np.log(df['TotalBsmtSF'] + 1)
    df['logGrLivArea']   = np.log(df['GrLivArea'] + 1)
    df['hasBsmt']        = (df['TotalBsmtSF'] > 0)
    df['hasGarage']      = (df['GarageArea'] > 0)


# In[17]:

train.head(3)


# In[18]:

test.head(3)


# In[19]:

train.to_csv('train_cleaned1.csv', index = False)
test.to_csv('test_cleaned1.csv', index = False)

