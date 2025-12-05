# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:47:11 2020

@author: Tejas

Approved AnalysisV2 - clean code till data cleaning. 
Timeseries generator and LSTMS's are added
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas_datareader.data as web
import datetime as dt
from peakdetect import peakdetect



threshLargeCap = 30 #$
beforeDays = 150 
afterDays = 150



# Read Dataframes
df = pd.read_csv('C:/Users/Tejas/OneDrive - Northeastern University/Finance/BioPharmHistorical_July5.csv')


## Take only Approved Drugs
df1 = df[df['phase'] == 'Approved']


## Create Yahoo Database
ticks = list(df1['ticker'].unique())
largeCap = []
for tick in ticks:   
    try:
        data = pd.read_csv('C:/Users/Tejas/OneDrive - Northeastern University/Finance/ApprovedAnalysis/YahooData/{}.csv'.format(tick))
    except:
        try:
            data = web.DataReader(tick, 'yahoo', dt.date(2000, 1, 1), dt.date(2020, 8, 8))
            data.reset_index(inplace = True)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
        
            data.to_csv('C:/Users/Tejas/OneDrive - Northeastern University/Finance/ApprovedAnalysis/YahooData/{}.csv'.format(tick), index = False)
        except:
            # if yahoo dosent have data then also remove that stock
            largeCap.append(tick)  
            pass
    
    if data.Close.iloc[-1] > threshLargeCap:
        print("Large Stock {} = {}".format(tick, data.Close.iloc[-1]))
        largeCap.append(tick)




### Filter all largeCaps and Unknowns
ticks1 = [x for x in ticks if x not in largeCap]


### Filter the dataset with only smallCap
df2 = df1[df1['ticker'].isin(ticks1)]
df2['Date'] = df2['Date'].apply(lambda x: dt.date(int(x.split('/')[2]), int(x.split('/')[0]), 
                                                  int(x.split('/')[1])))
df2.reset_index(drop = True, inplace = True)



### Add technical indicators

def MACD(df):
    
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean() #AKA Fast moving average
    LongEMA = df.Close.ewm(span=26, adjust=False).mean() #AKA Slow moving average
    
    macd = ShortEMA - LongEMA
    signal = macd.ewm(span=9, adjust=False).mean()
    
    df['Signal'] = signal
    df['MACD'] = macd
    df['MACD_diff'] = macd - signal
    
    return df


def kinematics(df):
    
    velocity = df.Close.diff()
    acc = velocity.diff()
    jerk = acc.diff()
    
    df['Velocity'] = velocity
    df['Acceleration'] = acc
    df['Jerk'] = jerk
    
    return df


def TEMA(df):
    #TEMA

    EMA = df['Close'].ewm(span = 6, adjust = False).mean()   
    TEMA = EMA.ewm(span = 6, adjust = False).mean()
    
    
    #Extract Tema from start to end date
    data['TEMA'] = TEMA
    data['Price_tema_diff'] = df['Close'] - TEMA
    return data
    

def RSI(df):
    delta = df.Close.diff(1)
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    avgGain = up.rolling(window = 14).mean()
    avgLoss = abs(down.rolling(window = 14).mean())
    
    RS = avgGain / avgLoss
    RSI = 100 - (100/(1+ RS))
    
    ### Convert RSI to binary column, where if RSI > 0.70, then RSI = 1
    RSI[RSI < 70] = 0    # First convert to zeros and then 1's
    RSI[RSI > 70] = 1
    
    
    df['RSI'] = RSI
    
    return df




### Seperate 100 day dataframes for LSTM data prep:
### Normalize close first itself
stocksList = []
df_all = pd.DataFrame()
for i in range(df2.shape[0]):
    catalyst_date = df2.Date.iloc[i]
    
    current_ticker = df2['ticker'].iloc[i]
    
    #clean data
    data = pd.read_csv('C:/Users/Tejas/OneDrive - Northeastern University/Finance/ApprovedAnalysis/YahooData/{}.csv'.format(current_ticker))
    data['Date'] = data['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
    

    # Index of catalyst date, take 40 trading days before and after
    ind = data[data['Date'] >= catalyst_date].index[0]
    indLow = ind - beforeDays
    indHigh = ind + afterDays
    
    d = TEMA(data.copy())
    d = MACD(d.copy())
    d = kinematics(d.copy())
    d = RSI(d.copy())
    
    current_df = d[(d.index >= indLow) & (d.index <= indHigh)]
    current_df.TEMA = current_df.TEMA.div(current_df.TEMA.iloc[0])#.mul(100)
    current_df.reset_index(drop = True, inplace = True)
    
    # If it has all the days data then add to another data  
    if current_df.shape[0] == (beforeDays + afterDays + 1):
        stocksList.append(current_df)
    else:
        pass




### create target labels 
num_of_pts = 0
for i in range(len(stocksList)):
    cdf = stocksList[i]
    cdf = cdf.fillna(cdf.mean(axis = 0))

    tema = cdf['TEMA']
    peaks = peakdetect(tema, lookahead=15) 
    
    
    ### Create a lag of one day and record the BUY and SELL Positions
    # 0 - Hold; 1- Sell; 2- Buy
    #targets = ['Hold']*len(tema)
    
    targets = [0]*len(tema)
    high = [x[0] for x in peaks[0]]
    low = [x[0] for x in peaks[1]]
    
    for ii in range(len(targets)):
        if ii in high:
            targets[ii-1] = 1#'Sell'
        if ii in low:
            targets[ii-1] = 2#'Buy'
            
    
    #targets = ['hold' for x in targets if x==0]        
    
    cdf['target'] = targets
    stocksList[i] = cdf
    num_of_pts += len(high) + len(low)
    
    '''
    ### Keep only the peaks and remove the rest
    TEMA_high = tema.copy()
    TEMA_low = tema.copy()
    
    for i in range(0, len(tema)):
        if i not in high: 
            TEMA_high[i] = np.nan
            
    for i in range(0, len(tema)):        
        if i not in low: 
            TEMA_low[i] = np.nan
    
    #num_of_pts += len(high) + len(low)
    
           
    ### plot the observations
    tema.plot(alpha = 0.35)
    cdf.Close.div(cdf.Close.iloc[0]).plot(label = 'HIGH')
    #SMA.plot(alpha = 0.6)
    plt.scatter(tema.index, TEMA_low, color = 'green', label = 'Buy', marker = '^', alpha = 1)
    plt.scatter(tema.index, TEMA_high, color = 'r', label = 'Sell', marker = 'v', alpha = 1)    
    plt.legend(loc = 'best')
    plt.show()
    '''
    



df_full = pd.concat(stocksList, axis=0)

X = df_full.drop(['Date', 'High', 'Low', 'Open', 'Close', 'Adj Close', 'target'], 1)
y = df_full['target']

from keras.utils import to_categorical
y = to_categorical(np.array(y))
'''
###### Undersampling
from imblearn.under_sampling import CondensedNearestNeighbour
# define the undersampling method
undersample = CondensedNearestNeighbour(n_neighbors=2)
# transform the dataset
X, y = undersample.fit_resample(X, y)
'''

X[['TEMA', 'Volume']] = np.log(X[['TEMA', 'Volume']])
X[['TEMA', 'Volume']].replace([0], 1, inplace = True)
X[['TEMA', 'Volume']].replace([-np.inf], 0, inplace = True)


from sklearn import model_selection
from sklearn import metrics

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, 
                                random_state = 40, shuffle = False)

print(X_train.shape, X_test.shape)



### Time Series generating
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

time_steps = 3
batch_size = 301
train_gen = TimeseriesGenerator(data = X_train, targets = y_train, sampling_rate = 1, length = time_steps, batch_size = batch_size)
test_gen = TimeseriesGenerator(data = X_test, targets = y_test, sampling_rate = 1, length = time_steps, batch_size = batch_size)
 


n_classes = 3
n_features = 10

#Always use tensorflow.keras instead of just keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, CuDNNLSTM
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras.backend import clear_session
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight 


model = Sequential()
model.add(CuDNNLSTM(1, input_shape=(time_steps, n_features), return_sequences=True))
#model.add(layers.Dropout(0.95))
model.add(CuDNNLSTM(1))
'''
model.add(CuDNNLSTM(6))
model.add(layers.Dropout(0.7))

model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dropout(0.7))
'''
model.add(Dense(n_classes, activation='softmax'))



early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min')
#sample_weights = compute_sample_weight('balanced', y_train)

### Declare an Optimizer - Adam Optimizer works well
opt = 'sgd'

# Compile model
## If 'sparse categorical crossentropy' is used as loss, then y_test and y_train need not be categorical, so use ytest n ytrain.
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

history = model.fit(train_gen,
          epochs=2,
          shuffle = False, 
          steps_per_epoch=len(train_gen),
          validation_data=test_gen,
          #sample_weights = sample_weights,
          class_weights = {0:0, 1:10000, 2:10000},
          callbacks = [early_stopping])


plt.plot(history.history['loss'])
plt.show()


predictions = model.predict_generator(test_gen)


y_actual = y_test[time_steps:]



# Many UNKNOWN AND UNSOLVED errors come up using this
#clear_session() # before training the model if you run this, it will throw an error
















































from keras.preprocessing.sequence import TimeseriesGenerator

time_steps = 2
features = cdf.drop(['Date', 'High', 'Low', 'Open', 'TEMA', 'target'], 1)
targets = cdf['target']
tsg = TimeseriesGenerator(data = features, targets = targets, sampling_rate = 1, length = time_steps, batch_size = 1)

tsg[1]







x = pd.DataFrame(tsg)
x.columns = ['features', 'targets']

from keras import layers, models        #### Using Keras Backend (tf backend)
from keras.utils import to_categorical  #### converts to categorical values to be suitable for neural net output layer

X = x['features']
y = x['targets']


## Removing the extra [] that tsg has created
X = X.apply(lambda x: x[0])
y = y.apply(lambda x: x[0])



y = to_categorical(np.array(y))



n_classes = 3
n_features = 11

X = pd.DataFrame(X)
X = X.apply(lambda x: x.reshape(1,time_steps, n_features))
#X = X.values.reshape(X.shape[0], time_steps, n_features)
X_test1 = X_test.reshape(X_test.shape[0], time_steps, n_features)   # Just make it 3D by adding 1 height layer 




from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.backend import clear_session

model = Sequential()
model.add(layers.LSTM(32, input_shape=(time_steps, n_features), return_sequences=False))
#model.add(layers.CuDNNLSTM(32))
#model.add(layers.Dropout(0.5))

#model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dropout(0.2))

model.add(layers.Dense(n_classes+1, activation='softmax'))




### Declare an Optimizer - Adam Optimizer works well
opt = 'Adam'

# Compile model
## If 'sparse categorical crossentropy' is used as loss, then y_test and y_train need not be categorical, so use ytest n ytrain.
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

history = model.fit(X,y)#,
          batch_size=14,
          epochs=50,
          validation_data=(X, y))


plt.plot(history.history['loss'])
plt.show()
clear_session()



np.reshape(cdf, (cdf.shape[0], time_steps, 17))
cdf.values.reshape(cdf.shape[0], time_steps, 17)




X = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]], [[10, 20, 30], [40, 50, 60], [70, 80, 90]]])
X = X.reshape(2, 3, 3)













