from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim, item_selection, select_sorted_items
from models import univariate_data, create_time_steps, show_plot, multivariate_data, multi_step_plot
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import csv
import time

TRAIN_SPLIT = 0
tf.random.set_seed(13)
STEP = 1

labels = ['timestamp', 'uni', 'multiS', 'multiM1', 'multiM2', 'multiM3', 'multiM4', 'multiM5']
def writeToCSV(filename, data, timestamp):
	with open('data/predictions/{}.csv'.format(filename), mode='w', newline='') as GE_data:
		GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		GE_writer.writerow(labels)  # write field names

		new_array = [timestamp]
		new_array.extend(data)
		GE_writer.writerow(new_array)


def appendToCSV(filename, data, timestamp):
	with open('data/predictions/{}.csv'.format(filename), mode='a', newline='') as GE_data:
		GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		new_array = [timestamp]
		new_array.extend(data)
		GE_writer.writerow(new_array)

def apply_univariate(df, item_to_predict, model, item_std, item_mean, past_history=30):

	df_newest_values = df.tail(past_history)[item_to_predict].values
	reshaped_values = np.reshape(df_newest_values, (past_history, 1))
	formatted_values = np.array([reshaped_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict(formatted_values)[0])
	
	return result

def apply_multivariate_single_step(df, item_to_predict, model, item_std, item_mean, past_history=30):

	df_newest_values = df.tail(past_history).values
	formatted_values = np.array([df_newest_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict(formatted_values)[0])
	
	return result

def apply_multivariate_multi_step(df, item_to_predict, model, item_std, item_mean, future_target=5, past_history=30):
	df_newest_values = df.tail(past_history).values
	formatted_values = np.array([df_newest_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict(formatted_values)[0])
	
	return result

def main():
	# Get the seconds since epoch
	current_timestamp = int(time.time())
	print("{} - predicting items".format(current_timestamp))

	model_types = ['uni', 'multiS', 'multiM']
	
	# SELECT ITEMS
	items_selected = item_selection(drop_percentage=0.5)
	items_to_predict = ['Amulet_of_strength', "Green_d'hide_vamb", 'Staff_of_fire', 'Zamorak_monk_top', 'Staff_of_air', \
			'Adamantite_bar', 'Zamorak_monk_bottom', 'Adamant_platebody', 'Runite_ore', 'Rune_scimitar', 'Rune_pickaxe', \
					'Rune_full_helm', 'Rune_kiteshield', 'Rune_2h_sword', 'Rune_platelegs', 'Rune_platebody', 'Old_school_bond']

	for item_to_predict in items_to_predict:
		# FEATURE EXTRACTION
		preprocessed_df = prepare_data(item_to_predict, items_selected, DATA_FOLDER="data/rsbuddy/")

		# FEATURE SELECTION & NORMALIZATION
		if not os.path.isfile('models/features/{}_{}_features.txt'.format(item_to_predict, model_types[0])):
			print ("Model for {} hasn't been created, please run models.py first.".format(item_to_predict))
			return
		specific_feature_list = []
		with open('models/features/{}_{}_features.txt'.format(item_to_predict, model_types[0]), 'r') as filehandle:
			specific_feature_list = json.load(filehandle)
		selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, \
			specific_features=specific_feature_list, number_of_features=len(specific_feature_list)-1)

		predictions = []
		for model_type in model_types:
			# LOADING AND APPLYING MODEL
			loaded_model = tf.keras.models.load_model('models/{}_{}_model.h5'.format(item_to_predict, model_type))

			if (model_type == 'uni'):
				result = apply_univariate(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)
			elif (model_type == 'multiS'):
				result = apply_multivariate_single_step(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)
			elif (model_type == 'multiM'):
				result = apply_multivariate_multi_step(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)
			else:
				print("Unrecognized model type.")
			
			predictions.extend(result)
		
		new_predictions = [int(i) for i in predictions]
		print('item: {}, pred: {}'.format(item_to_predict, new_predictions))
	
		if os.path.isfile('data/predictions/{}.csv'.format(item_to_predict)):
			appendToCSV(item_to_predict, new_predictions, current_timestamp)
		else:
			writeToCSV(item_to_predict, new_predictions, current_timestamp)


if __name__ == "__main__":
	main()