
# coding: utf-8

#### Titanic passengers survival prediction - EDA and Feature Engineering

# Load packages

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().magic(u'matplotlib inline')


# Import and inspect data

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


# We can see that there are 12 variables available in the data, *PassengerId* is just a primary key of the table and useless in prediction; *Survived* is our target variable and response in prediction; there are 5 numerical variables - *Pclass*, *Age*, *SibSp*, *Parch*, *Fare* and the rest are categorical variables, actually *Pclass* should be regarded as categorical

# We place the `train` and `test` into a list for data manipulation since every transformation should be applied on both of them. We can clearly see that family information can be extract directly: Whether come with family; number of family members.

# In[6]:

all_data = [train, test]
for df in all_data:
    df['Family'] = (df['SibSp'] + df['Parch'] == 0).map({True: 0, False: 1})
    df['num_family'] = df['SibSp'] + df['Parch']


# Then we find that there are titles listed in passengers' name, which are useful to indicate its *Sex* and *Age*, therefore we extract the title of passengers.

# In[7]:

# split name into last_name, title and first_name
for df in all_data:
    df[['last_name', 'title', 'first_name']] = df['Name'].str.split(',|\\.', expand = True).iloc[:, :3]


# In[8]:

# check all titles
train['title'].unique()


# In[9]:

train['title'].value_counts()


# Some titles actually point at same population, we convert them into same title:

# In[10]:

# merge title with same meaning
replace_title = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
for df in all_data:
    for old, new in replace_title.items():
        df['title'] = df['title'].str.replace(old, new)

train['title'].value_counts()


# and we find that some titles just have 1 or 2 records, we concate them called "other"

# In[11]:

# merge all rare titles into one category
rare_title = '(Lady)|(Capt)|(Don)|(Jonkheer)|(Sir)|(Col)|(Major)|(Rev)|(Dr)|(the Countess)'
for df in all_data:
    df['title'] = df['title'].str.replace(rare_title, 'other')

train['title'].value_counts()


# Next we intend to deal with *Age*, however we realise that there are many missing data

# In[12]:

# check missing data
for df in all_data:
    print(df.apply(lambda x: sum(x.isnull())))
    print('=================================')


# Oops, there are also several missing values in *Embarked* in `train` and *Fare* in `test`. We gonna impute them first since it's amount is small and easy to deal with To impute them. We explore the relationship between *Embarked* and *Fare* using boxplots.

# In[13]:

# check the relationship between Fare and Sex/Pclass/Embarked
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

# check the mean and median of groups (group by Sex - Pclass - Embarked)
test.loc[~idx_na_fare, ['Sex', 'Pclass', 'Embarked', 'Fare']].groupby(['Sex', 'Pclass', 'Embarked']).agg(['mean', 'median'])


# In[15]:

# fill the only missing Fare
test.loc[idx_na_fare, 'Fare'] = 7.9875


# In[16]:

# log-transform on Fare
for df in all_data:
    df['Fare'] = np.log(df['Fare'] + 1)

# check the relationship between Survived and Fare conditioning on Pclass and Embarked
mpl.rcParams['figure.figsize'] = 12, 9
p = sns.FacetGrid(train, row = 'Survived', col = 'Pclass')
p = p.map(sns.boxplot, 'Embarked', 'Fare')
plt.show()


# In[17]:

# check the record with missing Embarked
train.loc[train['Embarked'].isnull()]


# In[18]:

# fill missing Embarked with 'C'
train['Embarked'].fillna('C', inplace = True)
train['Embarked'].isnull().sum()


# In[19]:

# check the relationship between Age and Survived conditioning on Sex
p = sns.FacetGrid(train, col = 'Sex', size = 4)
p.map(sns.boxplot, 'Survived', 'Age')
plt.show()


# In[20]:

# check the relationship between Age and Survivred conditioning on Pclass
p = sns.FacetGrid(train, col = 'Pclass', size = 4)
p.map(sns.boxplot, 'Survived', 'Age')
plt.show()


# In[21]:

# check the relationship between Age and title conditioning on Sex
p = sns.FacetGrid(train, col = 'Sex', size = 4)
p.map(sns.boxplot, 'title', 'Age')
plt.show()


# In[22]:

# check the relationship between Age and Family conditioning on Survived
p = sns.FacetGrid(train, col = 'Survived', size = 4)
p.map(sns.boxplot, 'Family', 'Age')
plt.show()


# In[23]:

# check the relationship between Age and num_family
sns.boxplot(x = train['num_family'], y = train['Age'])
plt.show()


# In[24]:

# check the relationship between Age and Fare conditioning on Sex and Embarked
p = sns.FacetGrid(train, col = 'Embarked', row = 'Sex')
p.map(plt.scatter, 'Fare', 'Age', s = 18, alpha = 0.5)
plt.show()


# In[25]:

# check the mean and median Age group by Sex - Pclass - title - Family
train.groupby(['Sex', 'Pclass', 'title', 'Family'])[['Age']].agg(['mean', 'median'])


# In[26]:

# fill Age by median in group Sex - Pclass - title - Family
for df in all_data:
    df['imputed_age'] = df.groupby(['Sex', 'Pclass', 'title', 'Family'])['Age'].transform(lambda x: x.fillna(x.median()))
    
train.loc[train['Age'].isnull(), ['PassengerId', 'Age', 'imputed_age']].head()


# In[27]:

# compare the mean and median of Age with missing data and imputed Age
train.groupby(['Sex', 'Pclass', 'title', 'Family'])[['Age', 'imputed_age']].agg(['mean', 'median'])


# In[28]:

# check the relationship between Survived and adult conditioning on Sex
for df in all_data:
    df['adult'] = (df['Age'] < 18).map({True: 0, False: 1})

sns.factorplot(x = 'adult', y = 'Survived', hue = 'Sex', data = train, kind = 'bar', size = 5, aspect = 2)
plt.show()


# In[29]:

# check the distribution of imputed_age by male and female
train.groupby('Sex')['imputed_age'].plot.hist(alpha = 0.5, bins = 40)
plt.show()


# In[30]:

# check missing data again
for df in all_data:
    print(df.apply(lambda x: sum(x.isnull())))
    print('=================================')


# In[31]:

# decomposite Ticket: reformat it as letters + number of digits
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

# check the number of different last_name
pd.concat([train['last_name'], test['last_name']]).value_counts().count()


# In[33]:

# check most frequent last_name
last_name_set = pd.concat([train['last_name'], test['last_name']]).value_counts(ascending = True)
last_name_set.tail()


# In[34]:

# encode last_name
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

