from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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

	# Normalize
	uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
	uni_train_std = uni_data[:TRAIN_SPLIT].std()
	uni_data = (uni_data-uni_train_mean)/uni_train_std

	univariate_past_history = past_history
	univariate_future_target = 0

	x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
											univariate_past_history,
											univariate_future_target)
	x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
										univariate_past_history,
										univariate_future_target)

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

	simple_lstm_model.save('models/uni_model.h5')

# def apply_univariate(val_univariate, simple_lstm_model):
# 	#### Predict using the simple LSTM model
# 	for x, y in val_univariate.take(3):
# 		plot = show_plot([x[0].numpy(), y[0].numpy(),
# 						simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
# 		plot.show()

# 	#### Unnormalizing the data (so we can see actual prices in GP)
# 	def unnormalized(val):
# 		nonlocal uni_train_std, uni_train_mean
# 		return (val*uni_train_std) + uni_train_mean

# 	for x, y in val_univariate.take(2):
# 		plot = show_plot([unnormalized(x[0].numpy()), unnormalized(y[0].numpy()),
# 						unnormalized(simple_lstm_model.predict(x)[0])], 0, 'Simple LSTM model - unnormalized')
# 		plot.show()

  
def main():
	item_to_predict = 'Chaos_rune'
	items_selected = ['Chaos_rune', 'Death_rune', 'Nature_rune', 'Cosmic_rune', 'Law_rune']

	preprocessed_df = prepare_data(item_to_predict, items_selected)

	selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict)
	print(selected_df.head())

	# rfe_df = recursive_feature_elim(preprocessed_df, item_to_predict)

	univariate_rnn(selected_df, item_to_predict)

if __name__ == "__main__":
	main()