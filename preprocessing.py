from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# DATA_FOLDER = "data/osbuddy/excess/"
DATA_FOLDER = "data/rsbuddy/"
buy_average = pd.read_csv(DATA_FOLDER + "buy_average.csv")
buy_average = buy_average.set_index('timestamp')

buy_average = buy_average.drop_duplicates()

item_to_predict = 'Rune_scimitar'
items_selected = ['Rune_axe', 'Rune_2h_sword', 'Rune_scimitar', 'Rune_chainbody', 'Rune_full_helm', 'Rune_kiteshield']
df = buy_average[items_selected].replace(to_replace=0, method='ffill')
print(df.shape)

## Known finance features (MACD, RSI)

def moving_average_convergence(group, nslow=26, nfast=12):
    emaslow = group.ewm(span=nslow, min_periods=1).mean()
    emafast = group.ewm(span=nfast, min_periods=1).mean()
    result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    result = pd.DataFrame({'MACD': emafast-emaslow})
    return result

def moving_average(group, n=9):
    sma = group.rolling(n).mean()
    sma=sma.rename('SMA')
    return sma

def RSI(group, n=14):
    delta = group.diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean().abs()
    
    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    rsi=rsi.rename('RSI')
    return rsi

macd = moving_average_convergence(df[item_to_predict])
sma = moving_average(df[item_to_predict])
rsi = RSI(df[item_to_predict], 10)
finance_features = pd.concat([macd, rsi], axis=1)

## Fetched API features (buy quantity, sell price average)

sell_average = pd.read_csv(DATA_FOLDER + "sell_average.csv")
sell_average = sell_average.set_index('timestamp')
sell_average = sell_average.drop_duplicates()
sell_average = sell_average[items_selected].replace(to_replace=0, method='ffill')
sell_average.columns = [str(col) + '_sa' for col in sell_average.columns]

buy_quantity = pd.read_csv(DATA_FOLDER + "buy_quantity.csv")
buy_quantity = buy_quantity.set_index('timestamp')
buy_quantity = buy_quantity.drop_duplicates()
buy_quantity = buy_quantity[items_selected].replace(to_replace=0, method='ffill')
buy_quantity.columns = [str(col) + '_bq' for col in buy_quantity.columns]

sell_quantity = pd.read_csv(DATA_FOLDER + "sell_quantity.csv")
sell_quantity = sell_quantity.set_index('timestamp')
sell_quantity = sell_quantity.drop_duplicates()
sell_quantity = sell_quantity[items_selected].replace(to_replace=0, method='ffill')
sell_quantity.columns = [str(col) + '_sq' for col in sell_quantity.columns]

## Datetime properties

df['datetime'] = df.index
df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour

## Differentiated signal

tmp = df.copy()
tmp.index = pd.to_datetime(tmp.index)

slope = pd.Series(np.gradient(tmp[item_to_predict]), df.index, name='slope')
tmp = pd.concat([tmp, slope], axis=1)

## Appending features to main dataframe

df = pd.concat([df,finance_features, sell_average, buy_quantity, sell_quantity, slope], axis=1)
df = df.dropna()
print(df.shape)