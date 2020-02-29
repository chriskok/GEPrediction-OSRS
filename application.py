from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim, item_selection
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

TRAIN_SPLIT = 750
tf.random.set_seed(13)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i)
		# Reshape data from (history_size,) to (history_size, 1)
		data.append(np.reshape(dataset[indices], (history_size, 1)))
		labels.append(dataset[i+target_size])
	return np.array(data), np.array(labels)

def create_time_steps(length):
	time_steps = []
	for i in range(-length, 0, 1):
		time_steps.append(i)
	return time_steps

def show_plot(plot_data, delta, title):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, x in enumerate(plot_data):
		if i:
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	return plt

def apply_univariate(df, item_to_predict, model, item_std, item_mean, past_history=30, BATCH_SIZE=32):

	uni_data = df[item_to_predict]
	uni_data = uni_data.values

	univariate_past_history = past_history
	univariate_future_target = 0
	x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
										univariate_past_history,
										univariate_future_target)
	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	for x, y in val_univariate.take(2):
		# print(unnormalized(x[0].numpy()))
		plot = show_plot([unnormalized(x[0].numpy()), unnormalized(y[0].numpy()),
						unnormalized(model.predict(x)[0])], 0, 'Simple LSTM model - unnormalized')
		plot.show()


def main():
	# SELECT ITEMS
	items_selected = item_selection()
	# print(items_selected)
	item_to_predict = 'Death_rune'

	# FEATURE EXTRACTION
	preprocessed_df = prepare_data(item_to_predict, items_selected)

	# FEATURE SELECTION & NORMALIZATION
	if not os.path.isfile('models/features/{}_uni_features.txt'.format(item_to_predict)):
		print ("Model for {} hasn't been created please run models.py first".format(item_to_predict))
		return
	specific_feature_list = []
	with open('models/features/{}_uni_features.txt'.format(item_to_predict), 'r') as filehandle:
		specific_feature_list = json.load(filehandle)
	selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, specific_features=specific_feature_list)
	print(selected_df.head())

	# LOADING AND APPLYING MODEL
	loaded_model = tf.keras.models.load_model('models/{}_uni_model.h5'.format(item_to_predict))
	apply_univariate(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)

if __name__ == "__main__":
	main()