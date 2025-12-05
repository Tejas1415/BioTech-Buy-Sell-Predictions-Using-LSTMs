# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:03:03 2020

@author: Tejas

Stock Market and AI analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas_datareader.data as web
import datetime as dt


threshLargeCap = 30 #$
beforeDays = 50 
afterDays = 50



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



### For each of the selected events create a -60 days to +60 days database:
df_all = pd.DataFrame()
for i in range(20):#df2.shape[0]):
    catalyst_date = df2.Date.iloc[i]
    #start_date = catalyst_date - dt.timedelta(days = beforeDays)
    #end_date = catalyst_date + dt.timedelta(days = afterDays)
    
    current_ticker = df2['ticker'].iloc[i]
    
    #clean data
    data = pd.read_csv('C:/Users/Tejas/OneDrive - Northeastern University/Finance/ApprovedAnalysis/YahooData/{}.csv'.format(current_ticker))
    data['Date'] = data['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
    

    # Index of catalyst date, take 40 trading days before and after
    ind = data[data['Date'] >= catalyst_date].index[0]
    indLow = ind - beforeDays
    indHigh = ind + afterDays
    
    #Tema
    EMA = data['Close'].ewm(span = 6, adjust = False).mean()   
    tema = EMA.ewm(span = 6, adjust = False).mean()
    
    data['TEMA'] = tema
    
    current_df = data[(data.index >= indLow) & (data.index <= indHigh)]
    
    #Convert the TEMA for a normalized 100 range
    current_df.TEMA = current_df.TEMA.div(current_df.TEMA.iloc[0])#.mul(100)
    current_df.reset_index(inplace = True, drop = True)
    
    
    data['Close'].plot()
    #data['Close'].ewm(span = 6, adjust = False).mean().plot()
    current_df.TEMA.plot()
    plt.axvline(beforeDays, color = 'green', alpha = 0.4)
    plt.title('TEMA')
    plt.show()
 
   
    # If it has all the days data then add to another data
    if current_df.shape[0] == (beforeDays + afterDays + 1):
        df_all[current_ticker + '_{}'.format(i)] = current_df['TEMA']
        df_all.reset_index(inplace = True, drop = True)
    else:
        pass

   

## Plot all movements for all stocks
df_all.plot(legend = False)    

### Remove outliers - High gains. (12 events almost)
df_all1 = df_all[df_all.columns[df_all.max() < 2]]
df_all1.plot(legend = False)    
plt.show()
df_all1.mean(axis = 1).plot()
plt.show()







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








## Plotting something with normalized stocks
closePrice = data['Close'].iloc[:250]
closePrice.div(closePrice.iloc[0]).mul(100).plot()


### Printing the trend
import statsmodels.api as sm
sm.tsa.seasonal_decompose(data["Close"],freq=360).plot()



from peakdetect import peakdetect
closePrice = data['Close'].iloc[:250]
## EMA 
ma_period = 6
SMA = closePrice.rolling(window = ma_period-2).mean()
EMA = closePrice.ewm(span = ma_period-2, adjust = False).mean()   
EMA = EMA.ewm(span = ma_period-2, adjust = False).mean()


### By peakdetect lookahead
peaks = peakdetect(EMA, lookahead=15)
high = [x[0] for x in peaks[0]]
low = [x[0] for x in peaks[1]]


### Keep only the peaks and remove the rest
EMA_high = closePrice.copy()
EMA_low = closePrice.copy()

for i in range(0, len(EMA)):
    if i not in high: 
        EMA_high[i] = np.nan
        
for i in range(0, len(EMA)):        
    if i not in low: 
        EMA_low[i] = np.nan

       
### plot the observations
closePrice.plot(alpha = 0.35)
EMA.plot(label = 'EMA')
#SMA.plot(alpha = 0.6)
plt.scatter(EMA.index, EMA_low, color = 'green', label = 'Buy', marker = '^', alpha = 1)
plt.scatter(EMA.index, EMA_high, color = 'r', label = 'Sell', marker = 'v', alpha = 1)    
plt.legend(loc = 'best')
plt.show()










d = TEMA(data.copy())
d = MACD(d.copy())
d = kinematics(d.copy())
d = RSI(d.copy())

import ta
from ta import add_all_ta_features
data1 = add_all_ta_features(
    data.copy(), open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

data['Close1'] = ta.trend.EMAIndicator(data.Close)# window = 6)





