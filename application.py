from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim, item_selection
from models import univariate_data, create_time_steps, show_plot, multivariate_data, multi_step_plot
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

TRAIN_SPLIT = 750
tf.random.set_seed(13)
STEP = 1

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

	for x, y in val_univariate.take(5):
		# print(unnormalized(x[0].numpy()))
		plot = show_plot([unnormalized(x[0].numpy()), unnormalized(y[0].numpy()),
						unnormalized(model.predict(x)[0])], 0, 'Simple LSTM model - unnormalized')
		plot.show()

def apply_multivariate_single_step(df, item_to_predict, model, item_std, item_mean, past_history=30, BATCH_SIZE=32):
	dataset = df.values
	future_target = 1
	item_to_predict_index = df.columns.get_loc(item_to_predict)

	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAIN_SPLIT, None, past_history,
												future_target, STEP,
												single_step=True)
	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	for x, y in val_data_single.take(3):
		plot = show_plot([unnormalized(x[0][:, item_to_predict_index].numpy()), unnormalized(y[0].numpy()),
							unnormalized(model.predict(x)[0])], 1, 'Single Step Prediction - unnormalized')
		plot.show()

def apply_multivariate_multi_step(df, item_to_predict, model, item_std, item_mean, future_target=5, past_history=30, BATCH_SIZE=32):
	dataset = df.values
	item_to_predict_index = df.columns.get_loc(item_to_predict)

	x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAIN_SPLIT, None, past_history,
												future_target, STEP)

	val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
	val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean
	
	for x, y in val_data_multi.take(3):
		multi_step_plot(unnormalized(x[0].numpy()), unnormalized(y[0].numpy()), unnormalized(model.predict(x)[0]), item_to_predict_index)


def main():
	MODEL_TYPE = 'uni'
	
	# SELECT ITEMS
	items_selected = item_selection()
	# print(items_selected)
	item_to_predict = 'Old_school_bond'

	# FEATURE EXTRACTION
	preprocessed_df = prepare_data(item_to_predict, items_selected, DATA_FOLDER="data/newest/")

	# FEATURE SELECTION & NORMALIZATION
	if not os.path.isfile('models/features/{}_{}_features.txt'.format(item_to_predict, MODEL_TYPE)):
		print ("Model for {} hasn't been created, please run models.py first.".format(item_to_predict))
		return
	specific_feature_list = []
	with open('models/features/{}_{}_features.txt'.format(item_to_predict, MODEL_TYPE), 'r') as filehandle:
		specific_feature_list = json.load(filehandle)
	selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, \
		specific_features=specific_feature_list, number_of_features=len(specific_feature_list)-1)
	print(selected_df.head())
	print(selected_df.shape)

	# LOADING AND APPLYING MODEL
	loaded_model = tf.keras.models.load_model('models/{}_uni_model.h5'.format(item_to_predict))
	apply_univariate(selected_df, item_to_predict, loaded_model, pred_std, pred_mean, past_history=50)
	
	# loaded_model = tf.keras.models.load_model('models/{}_multiM_model.h5'.format(item_to_predict))
	# apply_multivariate_multi_step(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)

if __name__ == "__main__":
	main()