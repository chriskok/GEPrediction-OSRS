from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim, item_selection
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

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

def univariate_rnn(df, item_to_predict, past_history=30, BATCH_SIZE=32, BUFFER_SIZE=30, EVALUATION_INTERVAL=200, EPOCHS=10):
	uni_data = df[item_to_predict]
	uni_data = uni_data.values

	univariate_past_history = past_history
	univariate_future_target = 0

	x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
											univariate_past_history,
											univariate_future_target)
	x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
										univariate_past_history,
										univariate_future_target)

	train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
	train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	simple_lstm_model = tf.keras.models.Sequential([
		tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
		tf.keras.layers.Dense(1)
	])

	simple_lstm_model.compile(optimizer='adam', loss='mae')

	simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
						steps_per_epoch=EVALUATION_INTERVAL,
						validation_data=val_univariate, validation_steps=50)

	simple_lstm_model.save('models/{}_uni_model.h5'.format(item_to_predict))

	# open output file for writing
	with open('models/features/{}_uni_features.txt'.format(item_to_predict), 'w') as filehandle:
		json.dump(df.columns.values.tolist(), filehandle)

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

def apply_univariate_test(df, item_to_predict, model, item_std, item_mean, past_history=30, BATCH_SIZE=32):

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

# MULTIVARIATE PREDICTION FUNCTIONS
def multivariate_data(dataset, target, start_index, end_index, history_size,
					  target_size, step, single_step=False):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i, step)
		data.append(dataset[indices])

		if single_step:
			labels.append(target[i+target_size])
		else:
			labels.append(target[i:i+target_size])

	return np.array(data), np.array(labels)

def plot_train_history(history, title):
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()

	plt.show()

def multivariate_rnn_single(df, item_to_predict, past_history=100, BATCH_SIZE=32, BUFFER_SIZE=30, EVALUATION_INTERVAL=200, EPOCHS=10):
	dataset = df

	# # Normalize
	# dataset = features.values
	# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
	# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
	# dataset = (dataset-data_mean)/data_std

	future_target = 1
	STEP = 1

	item_to_predict_index = df.columns.get_loc(item_to_predict)

	x_train_single, y_train_single = multivariate_data(dataset, dataset[:, item_to_predict_index], 0,
													TRAIN_SPLIT, past_history,
													future_target, STEP,
													single_step=True)
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAIN_SPLIT, None, past_history,
												future_target, STEP,
												single_step=True)

	train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
	train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	single_step_model = tf.keras.models.Sequential()
	single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
	single_step_model.add(tf.keras.layers.Dense(1))
	single_step_model.add(tf.keras.layers.Dropout(0.2))
	single_step_model.add(tf.keras.layers.Dense(1))
	single_step_model.add(tf.keras.layers.Dropout(0.2))
	single_step_model.add(tf.keras.layers.Dense(1))

	single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae') #learning_rate=0.001

	single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
												steps_per_epoch=EVALUATION_INTERVAL,
												validation_data=val_data_single,
												validation_steps=50)

	plot_train_history(single_step_history,
					'Single Step Training and validation loss')

	#### Predict a single step future

	for x, y in val_data_single.take(3):
		plot = show_plot([x[0][:, item_to_predict_index].numpy(), y[0].numpy(),
							single_step_model.predict(x)[0]], 1,
						'Single Step Prediction')
		plot.show()


def main():
	# SELECT ITEMS
	items_selected = item_selection()
	# print(items_selected)
	item_to_predict = 'Chaos_rune'

	# FEATURE EXTRACTION
	preprocessed_df = prepare_data(item_to_predict, items_selected)

	# FEATURE SELECTION & NORMALIZATION
	selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict)
	# selected_df, pred_std, pred_mean = recursive_feature_elim(preprocessed_df, item_to_predict)
	print(selected_df.head())

	# # UNIVARIATE TRAINING AND SAVING MODEL
	# univariate_rnn(selected_df, item_to_predict)

	# # LOADING AND APPLYING MODEL
	# loaded_model = tf.keras.models.load_model('models/{}_uni_model.h5'.format(item_to_predict))
	# apply_univariate_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)

	print(selected_df.shape)
	print("columns with nan: {}".format(selected_df.columns[selected_df.isna().any()].tolist()))
	multivariate_rnn_single(selected_df, item_to_predict)

if __name__ == "__main__":
	main()