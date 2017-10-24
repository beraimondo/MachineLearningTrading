#import libraries

import numpy as np
import pandas as pd
import oandapy as opy
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')
path='.data'


#oanda platform credentials
accountId = 'MLfinclub'
token = '4d608abb7f46813447188432866d6698-7176a6a2d4443df441a471f2da62dd2f'

#connect to oanda
oanda = opy.API(environment='practice', access_token=token)

#retrieve data
d1 = '2017-10-01' #start date
d2 = '2017-10-10' #end date
instrument = 'GBP_USD'


# Download data step by step. Oanda doesn't permit to download more than 500 candles
dates = pd.date_range(start=d1, end=d2, freq='B') #freq=B business day

df = pd.DataFrame()
for i in range(0, len(dates) - 1):
    d1 = str(dates[i]).replace(' ', 'T')
    d2 = str(dates[i + 1]).replace(' ', 'T')
    try:
        data = oanda.get_history(instrument=instrument, start=d1, end=d2, granularity='H1')
        df = df.append(pd.DataFrame(data['candles']))
    except:
        pass

print(df)

#structuring the dataframe columns

"""1
df = pd.DataFrame(data["candles"]).set_index("time")
df.index = pd.DatetimeIndex(df.index)
"""

#set the index and eliminate useless columns
df.index = pd.DatetimeIndex(df['time'], tz='UTC')
del df['time']
del df['complete']


#add columns with daily highest high, lowest low and close
# (1 of 2) resample to Business day, find the key daily value and reindex to hourly
dfDay = pd.DataFrame()
dfDay['HH'] = df['highBid'].resample('B').max().shift(1).reindex(df.index).fillna(method="ffill") #ffill fills values forward, bfill backward
dfDay['LL'] = df['lowAsk'].resample('B').min().shift(1).reindex(df.index).fillna(method="ffill")
dfDay['C'] = df['closeBid'].resample('B').last().shift(1).reindex(df.index).fillna(method="ffill")

# (2 of 2) concatenate the dataframes
df = pd.concat([df, dfDay], axis=1)



# ML: preparing X and Y

# (1 of 2) this is X
for i in range(0, len(df)):
    if i == 0 or i == 1:
        df.ix[i, 'alfa01'], df.ix[i, 'alfa02'], df.ix[i, 'alfa03'], df.ix[i, 'alfa04'] = 1, 1, 1, 1
        df.ix[i, 'alfa05'], df.ix[i, 'alfa06'], df.ix[i, 'alfa07'], df.ix[i, 'alfa08'] = 1, 1, 1, 1
        df.ix[i, 'alfa09'], df.ix[i, 'alfa10'], df.ix[i, 'alfa11'], df.ix[i, 'alfa12'] = 1, 1, 1, 1
        df.ix[i, 'alfa13'], df.ix[i, 'alfa14'], df.ix[i, 'alfa15'], df.ix[i, 'alfa16'] = 1, 1, 1, 1

    else:
        df.ix[i, 'alfa01'] = (df.ix[i - 1, 'openAsk'] / df.ix[i - 2, 'openAsk'])
        df.ix[i, 'alfa02'] = (df.ix[i - 1, 'highAsk'] / df.ix[i - 2, 'openAsk'])
        df.ix[i, 'alfa03'] = (df.ix[i - 1, 'lowAsk'] / df.ix[i - 2, 'openAsk'])
        df.ix[i, 'alfa04'] = (df.ix[i - 1, 'closeAsk'] / df.ix[i - 2, 'openAsk'])

        df.ix[i, 'alfa05'] = (df.ix[i - 1, 'openAsk'] / df.ix[i - 2, 'highAsk'])
        df.ix[i, 'alfa06'] = (df.ix[i - 1, 'highAsk'] / df.ix[i - 2, 'highAsk'])
        df.ix[i, 'alfa07'] = (df.ix[i - 1, 'lowAsk'] / df.ix[i - 2, 'highAsk'])
        df.ix[i, 'alfa08'] = (df.ix[i - 1, 'closeAsk'] / df.ix[i - 2, 'highAsk'])

        df.ix[i, 'alfa09'] = (df.ix[i - 1, 'openAsk'] / df.ix[i - 2, 'lowAsk'])
        df.ix[i, 'alfa10'] = (df.ix[i - 1, 'highAsk'] / df.ix[i - 2, 'lowAsk'])
        df.ix[i, 'alfa11'] = (df.ix[i - 1, 'lowAsk'] / df.ix[i - 2, 'lowAsk'])
        df.ix[i, 'alfa12'] = (df.ix[i - 1, 'closeAsk'] / df.ix[i - 2, 'lowAsk'])

        df.ix[i, 'alfa13'] = (df.ix[i - 1, 'openAsk'] / df.ix[i - 2, 'closeAsk'])
        df.ix[i, 'alfa14'] = (df.ix[i - 1, 'highAsk'] / df.ix[i - 2, 'closeAsk'])
        df.ix[i, 'alfa15'] = (df.ix[i - 1, 'lowAsk'] / df.ix[i - 2, 'closeAsk'])
        df.ix[i, 'alfa16'] = (df.ix[i - 1, 'closeAsk'] / df.ix[i - 2, 'closeAsk'])


#function to discretize the values
def valueassign (a, b, c): # a=average, b=stnd dev, c= value
    if c > a-b and c < a+b:
        return 0
    else:
        if (c > (a-2*b) and c < (a-b)) or (c> (a+b) and c < (a+ 2*b)):
            return 1
        else:
            if (c > (a-3*b) and c < (a-2*b)) or (c> (a+2*b) and c < (a+ 3*b)):
                return 2
            else:
                return 3

#mean of the coefficients alfa
m01 = df['alfa01'].mean()
m02 = df['alfa02'].mean()
m03 = df['alfa03'].mean()
m04 = df['alfa04'].mean()

m05 = df['alfa05'].mean()
m06 = df['alfa06'].mean()
m07 = df['alfa07'].mean()
m08 = df['alfa08'].mean()

m09 = df['alfa09'].mean()
m10 = df['alfa10'].mean()
m11 = df['alfa11'].mean()
m12 = df['alfa12'].mean()

m13 = df['alfa13'].mean()
m14 = df['alfa14'].mean()
m15 = df['alfa15'].mean()
m16 = df['alfa16'].mean()


#standard deviation of the coefficients alfa
sda01 = df['alfa01'].std()
sda02 = df['alfa02'].std()
sda03 = df['alfa03'].std()
sda04 = df['alfa04'].std()

sda05 = df['alfa05'].std()
sda06 = df['alfa06'].std()
sda07 = df['alfa07'].std()
sda08 = df['alfa08'].std()

sda09 = df['alfa09'].std()
sda10 = df['alfa10'].std()
sda11 = df['alfa11'].std()
sda12 = df['alfa12'].std()

sda13 = df['alfa13'].std()
sda14 = df['alfa14'].std()
sda15 = df['alfa15'].std()
sda16 = df['alfa16'].std()

for i in range(0, len(df)):
    df.ix[i, 'alfa01'] = valueassign(m01, sda01, df.ix[i, 'alfa01'])
    df.ix[i, 'alfa02'] = valueassign(m02, sda02, df.ix[i, 'alfa02'])
    df.ix[i, 'alfa03'] = valueassign(m03, sda03, df.ix[i, 'alfa03'])
    df.ix[i, 'alfa04'] = valueassign(m04, sda04, df.ix[i, 'alfa04'])

    df.ix[i, 'alfa05'] = valueassign(m05, sda05, df.ix[i, 'alfa05'])
    df.ix[i, 'alfa06'] = valueassign(m06, sda06, df.ix[i, 'alfa06'])
    df.ix[i, 'alfa07'] = valueassign(m07, sda07, df.ix[i, 'alfa07'])
    df.ix[i, 'alfa08'] = valueassign(m08, sda08, df.ix[i, 'alfa08'])

    df.ix[i, 'alfa09'] = valueassign(m09, sda09, df.ix[i, 'alfa09'])
    df.ix[i, 'alfa10'] = valueassign(m10, sda10, df.ix[i, 'alfa10'])
    df.ix[i, 'alfa11'] = valueassign(m11, sda11, df.ix[i, 'alfa11'])
    df.ix[i, 'alfa12'] = valueassign(m12, sda12, df.ix[i, 'alfa12'])

    df.ix[i, 'alfa13'] = valueassign(m13, sda13, df.ix[i, 'alfa13'])
    df.ix[i, 'alfa14'] = valueassign(m14, sda14, df.ix[i, 'alfa14'])
    df.ix[i, 'alfa15'] = valueassign(m15, sda15, df.ix[i, 'alfa15'])
    df.ix[i, 'alfa16'] = valueassign(m16, sda16, df.ix[i, 'alfa16'])

# vectorizing X
from sklearn import metrics
dati = df[['alfa01', 'alfa02', 'alfa03', 'alfa04',
           'alfa05', 'alfa06', 'alfa07', 'alfa08',
           'alfa09', 'alfa10', 'alfa11', 'alfa12',
           'alfa13', 'alfa14', 'alfa15', 'alfa16']].values


# (2 of 2) this is Y (target vector)
for i in range(0, len(df)):
    if i == 0 :
        df.ix[i,'target'] = 1
    else:
        if df.ix[i, 'C'] > df.ix[i-1, 'HH'] or df.ix[i, 'C'] < df.ix[i-1, 'LL']:
            df.ix[i, 'target'] = 1
        else:
            df.ix[i,'target'] = 0


#print dataframe
desired_width = 350
pd.set_option('display.width', desired_width)
print(df)
print(df['target'])



# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dati, df.target)

# make predictions
expected = df.target
predicted = model.predict(dati)

# summarize the fit of the model
print(model)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# k-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(dati, df.target)

# make predictions
expected = df.target
predicted = model.predict(dati)

# summarize the fit of the model
print(model)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
