
# coding: utf-8

#### NYC taxi trip duration prediction - EDA and Feature Engineering

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# In[2]:

os.chdir('C:/Users/Bangda/Desktop/kaggle/ny-taxi')
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head()


# In[3]:

train.shape, test.shape


# In[4]:

# no missing data
all_data = [train, test]
for df in all_data:
    print(df.apply(lambda x: x.isnull().sum()))
    print("==================================")


# In[5]:

# calculate skewness and kurtosis of trip_duration and log(trip_duration)
print("skewness of raw trip_duration: {}".format(skew(train['trip_duration'])))
print("kurtosis of raw trip_duration: {}".format(kurtosis(train['trip_duration'])))
print("skewness of log-trip_duration: {}".format(skew(np.log(train['trip_duration']))))
print("kurtosis of log-trip_duration: {}".format(kurtosis(np.log(train['trip_duration']))))


# In[6]:

train['log_trip_duration'] = np.log(train['trip_duration'])
for df in all_data:
    df['pickup_longitude'] = np.abs(df['pickup_longitude'])
    df['dropoff_longitude'] = np.abs(df['dropoff_longitude'])


# In[7]:

# calculate the distance between two location based on latitude and longitude
def deg2rad(degree):
    return degree * np.pi / 180.0

def get_distance(long_x, lat_x, long_y, lat_y):
    R = 6371.0
    dlong = deg2rad(long_y - long_x)
    dlat = deg2rad(lat_y - lat_x)
    a = np.sin(dlat / 2.) ** 2 + np.cos(deg2rad(lat_x)) * np.cos(deg2rad(lat_y)) * np.sin(dlong / 2.) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
    return R * c


# In[8]:

# take an example, match with the result from google calculator
get_distance(train.loc[0, 'pickup_longitude'], train.loc[0, 'pickup_latitude'], 
             train.loc[0, 'dropoff_longitude'], train.loc[0, 'dropoff_latitude'])


# In[9]:

# vectorize the calculation function
get_distance = np.vectorize(get_distance)
for df in all_data:
    df['direct_distance'] = get_distance(np.array(df['pickup_longitude']), 
                                         np.array(df['pickup_latitude']), 
                                         np.array(df['dropoff_longitude']), 
                                         np.array(df['dropoff_latitude']))


# In[10]:

train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'direct_distance', 'log_trip_duration']].describe()


# In[11]:

# visualize pickup location
plt.figure(figsize = (12, 16))
plt.scatter(-train['pickup_longitude'].values, train['pickup_latitude'].values, alpha = 1./ 5., s = .3, c = '#22771F')
plt.xlim([-74.2, -73.68])
plt.ylim([40.52, 40.92])
plt.axis('off')
plt.show()


# In[12]:

# visualize dropoff location
plt.figure(figsize = (12, 15))
plt.scatter(-train['dropoff_longitude'].values, train['dropoff_latitude'].values, alpha = 1./ 5., s = .2, c = '#22771F')
plt.xlim([-74.2, -73.68])
plt.ylim([40.52, 40.92])
plt.grid()
plt.show()


# In[13]:

# extract month, day, hour, weekday
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
train['dropoff_month'] = train['dropoff_datetime'].apply(lambda x: x.month)
train['dropoff_day'] = train['dropoff_datetime'].apply(lambda x: x.day)
train['dropoff_hour'] = train['dropoff_datetime'].apply(lambda x: x.hour)
train['dropoff_weekday'] = train['dropoff_datetime'].apply(lambda x: x.weekday() + 1)

for df in all_data:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['pickup_day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['pickup_weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday() + 1)
    df['pickup_date'] = df['pickup_datetime'].apply(lambda x: x.date())


# In[14]:

train.groupby(['pickup_hour', 'dropoff_hour'])[['log_trip_duration']].agg(['mean', 'median', 'count'])


# In[15]:

# create business day variable
bday = pd.bdate_range(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-30'))
for df in all_data:
    df['is_bday'] = (df['pickup_datetime'].apply(lambda x: x.date() in bday)).map({True: 1, False: 0})


# In[16]:

# extreme weather event
weather_event = ['20160110', '20160113', '20160117', '20160123',
                 '20160205', '20160208', '20160215', '20160216',
                 '20160224', '20160225', '20160314', '20160315',
                 '20160328', '20160329', '20160403', '20160404',
                 '20160530', '20160628']
weather_event = pd.Series(pd.to_datetime(weather_event, format = '%Y%m%d')).dt.date


# In[17]:

for df in all_data:
    df['extreme_weather'] = df.pickup_date.isin(weather_event).map({True: 1, False: 0})


# In[18]:

# gather airport related trips
def is_jfk(longitude, latitude): 
    return (40.64 < latitude < 40.65) and (-73.79 < longitude < -73.76)
def is_ewr(longitude, latitude):
    return (40.68 < latitude < 40.70) and (-74.18 < longitude < -74.16)
def is_lga(longitude, latitude):
    return (40.77 < latitude < 40.78) and (-73.88 < longitude < -73.86)


# In[19]:

is_jfk = np.vectorize(is_jfk)
is_ewr = np.vectorize(is_ewr)
is_lga = np.vectorize(is_lga)

for df in all_data:
    df['is_jfk_pickup'] = is_jfk(-df['pickup_longitude'], df['pickup_latitude']).astype(np.float)
    df['is_ewr_pickup'] = is_ewr(-df['pickup_longitude'], df['pickup_latitude']).astype(np.float)
    df['is_lga_pickup'] = is_lga(-df['pickup_longitude'], df['pickup_latitude']).astype(np.float)
    df['is_jfk_dropoff'] = is_jfk(-df['dropoff_longitude'], df['dropoff_latitude']).astype(np.float)
    df['is_ewr_dropoff'] = is_ewr(-df['dropoff_longitude'], df['dropoff_latitude']).astype(np.float)
    df['is_lga_dropoff'] = is_lga(-df['dropoff_longitude'], df['dropoff_latitude']).astype(np.float)


# In[20]:

for df in all_data:
    pickup_hour = pd.get_dummies(df['pickup_hour'], drop_first = True)
    pickup_hour.columns = ['h' + str(i) for i in pickup_hour.columns]
    pickup_hour['h0'] = 0
    df[pickup_hour.columns] = pickup_hour


# In[21]:

import seaborn as sns


# In[22]:

# check the mean and median log-trip_duration by month
duration_by_month = train.groupby('pickup_month')[['log_trip_duration']].agg(['mean', 'median', 'count'])
plt.plot(duration_by_month.loc[:, 'log_trip_duration']['mean'].values,'o-r')
plt.plot(duration_by_month.loc[:, 'log_trip_duration']['median'].values, 'o-g')
plt.show()


# In[23]:

# check the mean and median log-trip_duration by day
duration_by_day = train.groupby('pickup_day')[['log_trip_duration']].agg(['mean', 'median', 'count'])
plt.plot(duration_by_day.loc[:, 'log_trip_duration']['mean'].values, 'o-r')
plt.plot(duration_by_day.loc[:, 'log_trip_duration']['median'].values, 'o-g')
plt.show()


# In[24]:

# check the mean and median log-trip_duration by weekday
duration_by_weekday = train.groupby('pickup_weekday')[['log_trip_duration']].agg(['mean', 'median', 'count'])
plt.plot(duration_by_weekday.loc[:, 'log_trip_duration']['mean'].values, 'o-r')
plt.plot(duration_by_weekday.loc[:, 'log_trip_duration']['median'].values, 'o-g')
plt.show()


# In[25]:

# check the mean and median log-trip_duration by hour
duration_by_hour = train.groupby('pickup_hour')[['log_trip_duration']].agg(['mean', 'median', 'count'])
plt.plot(duration_by_hour.loc[:, 'log_trip_duration']['mean'].values, 'o-r')
plt.plot(duration_by_hour.loc[:, 'log_trip_duration']['median'].values, 'o-g')
plt.show()


# In[26]:

# check the mean and median log-trip_duration by number of passengers
duration_by_passenger = train.groupby('passenger_count')[['log_trip_duration']].agg(['mean', 'median', 'count'])
plt.plot(duration_by_passenger.loc[2:6, 'log_trip_duration']['mean'], 'o-r')
plt.plot(duration_by_passenger.loc[2:6, 'log_trip_duration']['median'], 'o-g')
plt.xticks(np.arange(2, 7))
plt.show()


# In[27]:

# create polynomial features
for df in all_data:
    df['pickup_weekdaysq'] = df['pickup_weekday'] ** 2
    df['pickup_daysq'] = df['pickup_day'] ** 2
    df['pickup_daycb'] = df['pickup_day'] ** 3
    df['pickup_dayqd'] = df['pickup_day'] ** 4
    df['passenger_countsq'] = df['passenger_count'] ** 2


# In[28]:

test.head()


# In[29]:

# store_and_fwd_flag N/Y
bplot = sns.boxplot(x = 'store_and_fwd_flag', y = 'log_trip_duration', data = train)
plt.show()


# In[30]:

# check the distribution of y, log(y)
trip_hist = train[['trip_duration', 'log_trip_duration']].hist(layout = (2, 1), bins = 80)
trip_hist[1, 0].set_xlim((0, 20000))
plt.show()


# In[31]:

train['log_trip_duration'].describe()


# In[32]:

# check the cumulative distribution of log(y)
quantile_log_trip_duration = train['log_trip_duration'].quantile(np.arange(0, 1.001, .001))
plt.plot(quantile_log_trip_duration.values[:-1], quantile_log_trip_duration.index[:-1], 'o-')
plt.show()
print('first 5 values: {}'.format(quantile_log_trip_duration.head().values))
print('last 5 values: {}'.format(quantile_log_trip_duration.tail().values))


# In[33]:

# 6-sigma principle
lower = train['log_trip_duration'].mean() - 3 * train['log_trip_duration'].std()
upper = train['log_trip_duration'].mean() + 3 * train['log_trip_duration'].std()
print('6-sigma range of trip_duration: {}'.format([(np.exp(lower) - 1) / 3600., (np.exp(upper) - 1) / 3600.]))


# In[34]:

# too long / short trip duration might be incorrect
train['abnormal_trip'] = np.logical_or(train.log_trip_duration < 1.945, train.log_trip_duration > 11.35)
train['abnormal_trip'] = train['abnormal_trip'].map({True: 1, False: 0})
train.loc[train['abnormal_trip'] == 1, ['pickup_datetime', 'dropoff_datetime', 'direct_distance', 'trip_duration']].head()


# In[35]:

train.abnormal_trip.value_counts() / train.shape[0]


# In[36]:

# calculate average speed during the trip
train['avg_speed'] = np.divide(train['direct_distance'], train['trip_duration'] / 3600.)
train['avg_speed'].describe()


# In[37]:

# 6-sigma principle
train['log_avg_speed'] = np.log(train['avg_speed'] + 1)
lower = train['log_avg_speed'].mean() - 3 * train['log_avg_speed'].std()
upper = train['log_avg_speed'].mean() + 3 * train['log_avg_speed'].std()
print('6-sigma range of average speed: {}'.format([np.exp(lower) - 1, np.exp(upper) - 1]))


# In[38]:

quantile_avg_speed = train['avg_speed'].quantile(np.arange(0, 1.001, .001))
plt.plot(quantile_avg_speed.values[:-1], quantile_avg_speed.index[:-1], 'o-')
plt.show()
print('first 5 values: {}'.format(quantile_avg_speed.head().values))
print('last 5 values: {}'.format(quantile_avg_speed.tail().values))


# In[39]:

# extremely high speed might be incorrect
train['abnormal_speed'] = train.avg_speed > np.exp(upper) - 1
train['abnormal_speed'] = train['abnormal_speed'].map({True: 1, False: 0})
train.loc[train['abnormal_speed'] == 1, ['pickup_datetime', 'dropoff_datetime', 'direct_distance', 'trip_duration', 'avg_speed', 'abnormal_trip']].head()


# In[40]:

train.abnormal_speed.value_counts() / train.shape[0]


# In[41]:

# lower avg_speed might be traffic jam
train['maybe_jam'] = np.logical_and(train.avg_speed < np.exp(lower) - 1, train.log_trip_duration < upper)
train['maybe_jam'] = train['maybe_jam'].map({True: 1, False: 0})
train.loc[train.maybe_jam == 1, ['pickup_datetime', 'dropoff_datetime', 
                                 'pickup_longitude', 'pickup_latitude',
                                 'dropoff_longitude', 'dropoff_latitude',
                                 'direct_distance', 'trip_duration', 'avg_speed', 'abnormal_trip']].head()


# In[42]:

# set abnormal observations
train['abnormal'] = train['abnormal_trip'] + train['abnormal_speed']


# In[43]:

# add neighbors info
# from https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/38220
train_neighbors = pd.read_csv('C:/Users/bangda/desktop/kaggle/ny-taxi/external-data/train_neighbors.csv')
test_neighbors  = pd.read_csv('C:/Users/bangda/desktop/kaggle/ny-taxi/external-data/test_neighbors.csv')
train = pd.concat([all_data[0], train_neighbors], axis = 'columns')
test  = pd.concat([all_data[1], test_neighbors], axis = 'columns')
all_data = [train, test]


# In[44]:

train.iloc[:, -8:].head()


# In[45]:

train.pickup_boro.value_counts()


# In[46]:

train.pickup_boro_code.value_counts()


# In[47]:

train.pickup_boro.isnull().sum(), train.dropoff_boro.isnull().sum(), test.pickup_boro.isnull().sum(), test.dropoff_boro.isnull().sum()


# In[48]:

train.pickup_neighborhood_code.value_counts().shape, train.dropoff_neighborhood_code.value_counts().shape, test.pickup_neighborhood_code.value_counts().shape, test.dropoff_neighborhood_code.value_counts().shape


# In[49]:

train.pickup_neighborhood_code.isnull().sum(), test.pickup_neighborhood_code.isnull().sum()


# In[50]:

# fill missing data (trip out of NYC)
for df in all_data:
    df['pickup_boro_code'].fillna(value = 6., inplace = True)
    df['dropoff_boro_code'].fillna(value = 6., inplace = True)
    df['pickup_neighborhood_code'].fillna(value = 'out_ny', inplace = True)
    df['dropoff_neighborhood_code'].fillna(value = 'out_ny', inplace = True)


# In[51]:

for df in all_data:
    df.drop(['pickup_boro', 'dropoff_boro', 'pickup_neighborhood_name', 'dropoff_neighborhood_name'], inplace = True, axis = 'columns')


# In[52]:

train.head()


# In[53]:

# create dummy variable for hour
for df in all_data:
    pickup_hour = pd.get_dummies(df['pickup_hour'], drop_first = True)
    pickup_hour.columns = ['h' + str(i) for i in pickup_hour.columns]
    pickup_hour['h0'] = 0
    df[pickup_hour.columns] = pickup_hour


# In[54]:

# create dummy variable for neighbors
for df in all_data:
    pickup_boro_code = pd.get_dummies(df['pickup_boro_code'], drop_first = True)
    pickup_boro_code.columns = ['pn' + str(i) for i in pickup_boro_code.columns]
    pickup_boro_code['pn1'] = 0
    df[pickup_boro_code.columns] = pickup_boro_code
    dropoff_boro_code = pd.get_dummies(df['dropoff_boro_code'], drop_first = True)
    dropoff_boro_code.columns = ['dn' + str(i) for i in dropoff_boro_code.columns]
    dropoff_boro_code['dn1'] = 0
    df[dropoff_boro_code.columns] = dropoff_boro_code    


# In[55]:

all_pickup_neighborhood_code = np.union1d(train.pickup_neighborhood_code.values, test.pickup_neighborhood_code.values)
all_dropoff_neighborhood_code = np.union1d(train.pickup_neighborhood_code.values, test.pickup_neighborhood_code.values)
np.union1d(all_pickup_neighborhood_code, all_dropoff_neighborhood_code)


# In[56]:

train_test = train.loc[:, ['pickup_neighborhood_code', 'dropoff_neighborhood_code']].append(test.loc[:, ['pickup_neighborhood_code', 'dropoff_neighborhood_code']])
train_test_pickup_neighborhood_code = pd.get_dummies(train_test['pickup_neighborhood_code'])
train_test_dropoff_neighborhood_code = pd.get_dummies(train_test['dropoff_neighborhood_code'])


# In[57]:

# split into train and test
train[train_test_pickup_neighborhood_code.columns] = train_test_pickup_neighborhood_code.iloc[:1458644]
test[train_test_pickup_neighborhood_code.columns] = train_test_pickup_neighborhood_code.iloc[1458644:]


# In[58]:

train[train_test_dropoff_neighborhood_code.columns] = train_test_dropoff_neighborhood_code.iloc[:1458644]
test[train_test_dropoff_neighborhood_code.columns] = train_test_dropoff_neighborhood_code.iloc[1458644:]


# In[59]:

del train_test_pickup_neighborhood_code, train_test_dropoff_neighborhood_code, train_neighbors, test_neighbors, train_test


# In[60]:

train.shape, test.shape


# In[61]:

# add weather info
# from
weather = pd.read_csv('C:/Users/bangda/desktop/kaggle/ny-taxi/external-data/Weather.csv', index_col = 0)
weather = weather.loc['2016-01-01':'2016-07-01']
weather.tail()


# In[62]:

weather['pickup_date'] = pd.to_datetime(weather.index)
weather['date'] = weather['pickup_date'].dt.strftime('%Y%m%d')
weather['hour'] = weather['pickup_date'].dt.strftime('%H')
weather['date_hour'] = weather['date'] + weather['hour']
weather = weather.reset_index()


# In[63]:

weather.head()


# In[64]:

weather = weather[['date_hour', 'tempi', 'hum', 'wspdi', 'precipm']]


# In[65]:

# create match id
for df in all_data:
    df['date_hour'] = df['pickup_datetime'].dt.strftime('%Y%m%d') + df['pickup_datetime'].dt.strftime('%H')


# In[66]:

# join tables
train = train.join(weather, on = 'date_hour', how = 'left', lsuffix = '_train', rsuffix = '_weather')
test  = test.join(weather, on = 'date_hour', how = 'left', lsuffix = '_test', rsuffix = '_weather')


# In[67]:

del weather
all_data = [train, test]
for df in all_data:
    df['precipm'].fillna(0.0, inplace = True)
    df['wspdi'].fillna(0.0, inplace = True)
    df['hum'].fillna(0.0, inplace = True)
    df['tempi'].fillna(0.0, inplace = True)


# In[68]:

train.columns.values


# In[69]:

del train['date_hour_train']
del test['date_hour_test']

for df in all_data:
    del df['date_hour_weather']


# In[70]:

# gather and drop some variables
for df in all_data:
    del df['pickup_boro_code']
    del df['pickup_neighborhood_code']
    del df['dropoff_boro_code']
    del df['dropoff_neighborhood_code']
    df['jfk_trip'] = df['is_jfk_pickup'] + df['is_jfk_dropoff']
    del df['is_jfk_pickup']
    del df['is_jfk_dropoff']
    df['ewr_trip'] = df['is_ewr_pickup'] + df['is_ewr_dropoff']
    del df['is_ewr_pickup']
    del df['is_ewr_dropoff']
    df['lga_trip'] = df['is_lga_pickup'] + df['is_lga_dropoff']
    del df['is_lga_pickup']
    del df['is_lga_dropoff']
    del df['is_bday']
    df['pickup_weekdaycb'] = df['pickup_weekday'] ** 3
    df['passenger_countcb'] = df['passenger_count'] ** 3
    df['out_ny'] = 0


# In[71]:

del train['dropoff_month']
del train['dropoff_day']
del train['dropoff_hour']
del train['dropoff_weekday']


# In[72]:

test.columns.values


# In[73]:

train.shape, test.shape


# In[74]:

predictors = [
       'passenger_count', 
       'direct_distance',
       'pickup_month', 'pickup_day', 'pickup_weekday',
       'extreme_weather', 'h1', 'h2', 'h3', 'h4', 'h5',
       'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15',
       'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h0',
       'pickup_weekdaysq', 'pickup_daysq', 'pickup_daycb', 'pickup_dayqd',
       'passenger_countsq', 
       'pn2.0', 'pn3.0',
       'pn4.0', 'pn5.0', 'pn6.0', 'pn1', 'dn2.0', 'dn3.0', 'dn4.0',
       'dn5.0', 'dn6.0', 'dn1', 'BK09', 'BK17', 'BK19', 'BK21', 'BK23',
       'BK25', 'BK26', 'BK27', 'BK28', 'BK29', 'BK30', 'BK31', 'BK32',
       'BK33', 'BK34', 'BK35', 'BK37', 'BK38', 'BK40', 'BK41', 'BK42',
       'BK43', 'BK44', 'BK45', 'BK46', 'BK50', 'BK58', 'BK60', 'BK61',
       'BK63', 'BK64', 'BK68', 'BK69', 'BK72', 'BK73', 'BK75', 'BK76',
       'BK77', 'BK78', 'BK79', 'BK81', 'BK82', 'BK83', 'BK85', 'BK88',
       'BK90', 'BK91', 'BK93', 'BK95', 'BK96', 'BK99', 'BX01', 'BX03',
       'BX05', 'BX06', 'BX07', 'BX08', 'BX09', 'BX10', 'BX13', 'BX14',
       'BX17', 'BX22', 'BX26', 'BX27', 'BX28', 'BX29', 'BX30', 'BX31',
       'BX33', 'BX34', 'BX35', 'BX36', 'BX37', 'BX39', 'BX40', 'BX41',
       'BX43', 'BX44', 'BX46', 'BX49', 'BX52', 'BX55', 'BX59', 'BX62',
       'BX63', 'BX75', 'BX98', 'BX99', 'MN01', 'MN03', 'MN04', 'MN06',
       'MN09', 'MN11', 'MN12', 'MN13', 'MN14', 'MN15', 'MN17', 'MN19',
       'MN20', 'MN21', 'MN22', 'MN23', 'MN24', 'MN25', 'MN27', 'MN28',
       'MN31', 'MN32', 'MN33', 'MN34', 'MN35', 'MN36', 'MN40', 'MN50',
       'MN99', 'QN01', 'QN02', 'QN03', 'QN05', 'QN06', 'QN07', 'QN08',
       'QN10', 'QN12', 'QN15', 'QN17', 'QN18', 'QN19', 'QN20', 'QN21',
       'QN22', 'QN23', 'QN25', 'QN26', 'QN27', 'QN28', 'QN29', 'QN30',
       'QN31', 'QN33', 'QN34', 'QN35', 'QN37', 'QN38', 'QN41', 'QN42',
       'QN43', 'QN44', 'QN45', 'QN46', 'QN47', 'QN48', 'QN49', 'QN50',
       'QN51', 'QN52', 'QN53', 'QN54', 'QN55', 'QN56', 'QN57', 'QN60',
       'QN61', 'QN62', 'QN63', 'QN66', 'QN68', 'QN70', 'QN71', 'QN72',
       'QN76', 'QN98', 'QN99', 'SI01', 'SI05', 'SI07', 'SI08', 'SI12',
       'SI14', 'SI22', 'SI25', 'SI28', 'SI32', 'SI35', 'SI36', 'SI37',
       'SI54', 'out_ny', 'SI11', 'SI24', 'SI45', 'SI48', 'tempi', 'hum',
       'wspdi', 'precipm', 'jfk_trip', 'ewr_trip', 'lga_trip',
       'pickup_weekdaycb', 'passenger_countcb']


# In[75]:

len(predictors)


# In[76]:

train.to_csv('train_featured.csv', index = False)
test.to_csv('test_featured.csv', index = False)


# In[77]:

import gc
gc.collect()

