
# coding: utf-8

#### Titanic passengers survival prediction - EDA and Feature Engineering

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().magic(u'matplotlib inline')


# In[2]:

os.chdir('C:/Users/Bangda/Desktop/kaggle/titanic')
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head(3)


# In[3]:

train.shape, test.shape


# In[4]:

test['Survived'] = np.nan


# In[5]:

train.info()


# In[6]:

all_data = [train, test]
for df in all_data:
    df['Family'] = (df['SibSp'] + df['Parch'] == 0).map({True: 0, False: 1})
    df['num_family'] = df['SibSp'] + df['Parch']


# In[7]:

for df in all_data:
    df[['last_name', 'title', 'first_name']] = df['Name'].str.split(',|\\.', expand = True).iloc[:, :3]


# In[8]:

train['title'].unique()


# In[9]:

train['title'].value_counts()


# In[10]:

replace_title = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
for df in all_data:
    for old, new in replace_title.items():
        df['title'] = df['title'].str.replace(old, new)

train['title'].value_counts()


# In[11]:

rare_title = '(Lady)|(Capt)|(Don)|(Jonkheer)|(Sir)|(Col)|(Major)|(Rev)|(Dr)|(the Countess)'
for df in all_data:
    df['title'] = df['title'].str.replace(rare_title, 'other')

train['title'].value_counts()


# In[12]:

for df in all_data:
    print(df.apply(lambda x: sum(x.isnull())))
    print('=================================')


# In[13]:

mpl.rcParams['figure.figsize'] = 12, 9
idx_na_fare = test['Fare'].isnull()
plt.subplot(2, 2, 1)
sns.boxplot(x = test.loc[~idx_na_fare, 'Sex'], y = np.log(test.loc[~idx_na_fare, 'Fare'] + 1))
plt.subplot(2, 2, 2)
sns.boxplot(x = test.loc[~idx_na_fare, 'Pclass'], y = np.log(test.loc[~idx_na_fare, 'Fare'] + 1))
plt.subplot(2, 2, 3)
sns.boxplot(x = test.loc[~idx_na_fare, 'Embarked'], y = np.log(test.loc[~idx_na_fare, 'Fare'] + 1))
plt.show()


# In[14]:

test.loc[~idx_na_fare, ['Sex', 'Pclass', 'Embarked', 'Fare']].groupby(['Sex', 'Pclass', 'Embarked']).agg(['mean', 'median'])


# In[15]:

test.loc[idx_na_fare, 'Fare'] = 7.9875


# In[16]:

for df in all_data:
    df['Fare'] = np.log(df['Fare'] + 1)

mpl.rcParams['figure.figsize'] = 12, 9
p = sns.FacetGrid(train, row = 'Survived', col = 'Pclass')
p = p.map(sns.boxplot, 'Embarked', 'Fare')
plt.show()


# In[17]:

train.loc[train['Embarked'].isnull()]


# In[18]:

train['Embarked'].fillna('C', inplace = True)
train['Embarked'].isnull().sum()


# In[19]:

p = sns.FacetGrid(train, col = 'Sex', size = 4)
p.map(sns.boxplot, 'Survived', 'Age')
plt.show()


# In[20]:

p = sns.FacetGrid(train, col = 'Pclass', size = 4)
p.map(sns.boxplot, 'Survived', 'Age')
plt.show()


# In[21]:

p = sns.FacetGrid(train, col = 'Sex', size = 4)
p.map(sns.boxplot, 'title', 'Age')
plt.show()


# In[22]:

p = sns.FacetGrid(train, col = 'Survived', size = 4)
p.map(sns.boxplot, 'Family', 'Age')
plt.show()


# In[23]:

sns.boxplot(x = train['num_family'], y = train['Age'])
plt.show()


# In[24]:

p = sns.FacetGrid(train, col = 'Embarked', row = 'Sex')
p.map(plt.scatter, 'Fare', 'Age', s = 18, alpha = 0.5)
plt.show()


# In[25]:

train.groupby(['Sex', 'Pclass', 'title', 'Family'])[['Age']].agg(['mean', 'median'])


# In[26]:

for df in all_data:
    df['imputed_age'] = df.groupby(['Sex', 'Pclass', 'title', 'Family'])['Age'].transform(lambda x: x.fillna(x.median()))
    
train.loc[train['Age'].isnull(), ['PassengerId', 'Age', 'imputed_age']].head()


# In[27]:

train.groupby(['Sex', 'Pclass', 'title', 'Family'])[['Age', 'imputed_age']].agg(['mean', 'median'])


# In[28]:

for df in all_data:
    df['adult'] = (df['Age'] < 18).map({True: 0, False: 1})

sns.factorplot(x = 'adult', y = 'Survived', hue = 'Sex', data = train, kind = 'bar', size = 5, aspect = 2)
plt.show()


# In[29]:

train.groupby('Sex')['imputed_age'].plot.hist(alpha = 0.5, bins = 40)
plt.show()


# In[30]:

for df in all_data:
    print(df.apply(lambda x: sum(x.isnull())))
    print('=================================')


# In[31]:

def findLetterLength(ticket):
    return len(re.compile('[A-Z]+').findall(ticket))

def findDigitLength(ticket):
    all_digits = ''.join(re.compile('[0-9]+').findall(ticket)) 
    return len(all_digits)

def reTicket(ticket):
    return ''.join(re.compile('[A-Z]+').findall(ticket)) + str(findDigitLength(ticket))

for df in all_data:
    df['ticket_letter_length'] = df['Ticket'].apply(findLetterLength)
    df['ticket_digit_length'] = df['Ticket'].apply(findDigitLength)
    df['re_ticket'] = df['Ticket'].apply(reTicket)

train.groupby('re_ticket')[['Survived']].agg(['mean', 'count'])


# In[32]:

pd.concat([train['last_name'], test['last_name']]).value_counts().count()


# In[33]:

last_name_set = pd.concat([train['last_name'], test['last_name']]).value_counts(ascending = True)
last_name_set.tail()


# In[34]:

last_name_ref = dict(zip(last_name_set.index, np.arange(1, len(last_name_set) + 1)))
for df in all_data:
    df['family_id'] = df['last_name'].map(last_name_ref)


# In[35]:

train.head(3)


# In[36]:

test.head(3)


# In[37]:

train.to_csv('train_cleaned.csv', index = False)
test.to_csv('test_cleaned.csv', index = False)

