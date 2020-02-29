from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import pandas as pd

from sklearn import datasets
from sklearn.feature_selection import RFE, f_regression, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# DATA_FOLDER = "data/osbuddy/excess/"
DATA_FOLDER = "data/rsbuddy/"

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

def prepare_data(item_to_predict, items_selected):
	buy_average = pd.read_csv(DATA_FOLDER + "buy_average.csv")
	buy_average = buy_average.set_index('timestamp')
	buy_average = buy_average.drop_duplicates()
	df = buy_average[items_selected].replace(to_replace=0, method='ffill')

	## Known finance features (MACD, RSI)
	macd = moving_average_convergence(df[item_to_predict])
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

	return df

# FEATURE SELECTION FUNCTIONS

def regression_f_test(input_df, item_to_predict, print_scores=False):
	features = input_df.drop(['datetime'], axis=1).copy()

	# normalize dataset
	features_std = features.std()
	features_mean = features.mean()
	dataset=(features-features_mean)/features_std
		
	X = dataset.drop([item_to_predict], axis=1)
	y = dataset[item_to_predict]

	# define feature selection
	fs = SelectKBest(score_func=f_regression, k=7)
	# apply feature selection
	fs.fit_transform(X, y)

	# Get scores for each of the columns
	scores = fs.scores_
	if print_scores:
		for idx, col in enumerate(X.columns): 
			print("feature: {: >20} \t score: {: >10}".format(col, round(scores[idx],5)))

	# Get columns to keep and create new dataframe with those only
	cols = fs.get_support(indices=True)
	features_df_new = X.iloc[:,cols]
	
	# print('std: {}, mean: {}'.format(features_std[item_to_predict], features_mean[item_to_predict]))
	return pd.concat([features_df_new, y], axis=1), features_std[item_to_predict], features_mean[item_to_predict]

def recursive_feature_elim(input_df, item_to_predict):
	features = input_df.drop(['datetime'], axis=1).copy()

	# normalize dataset
	features_std = features.std()
	features_mean = features.mean()
	dataset=(features-features_mean)/features_std

	X = dataset.drop([item_to_predict], axis=1)
	y = dataset[item_to_predict]

	# perform feature selection
	rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 7)
	fit = rfe.fit(X, y)
	# report selected features
	print('Selected Features:')
	names = dataset.drop([item_to_predict], axis=1).columns.values
	selected_features = []
	for i in range(len(fit.support_)):
		if fit.support_[i]:
			selected_features.append(names[i])

	return pd.concat([X[selected_features], y], axis=1), features_std[item_to_predict], features_mean[item_to_predict]

# Unnormalizing the data (so we can see actual prices in GP)
def unnormalized(val, std, mean):
	return (val*std) + mean

def main():
	item_to_predict = 'Rune_scimitar'
	items_selected = ['Rune_axe', 'Rune_2h_sword', 'Rune_scimitar', 'Rune_chainbody', 'Rune_full_helm', 'Rune_kiteshield']

	preprocessed_df = prepare_data(item_to_predict, items_selected)
	print(preprocessed_df.head())

	selected_data, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict)
	print(selected_data.head())
	print(selected_data.shape)
	# print(unnormalized(selected_data[item_to_predict], pred_std, pred_mean))

if __name__ == "__main__":
	main()