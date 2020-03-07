from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
import json
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route('/')
def index():
	# items_predicted = ['Old_school_bond', 'Rune_platebody', 'Adamant_platebody', "Red_spiders'_eggs", 'Ruby_necklace', 'Amulet_of_strength']
	# items_predicted = ["Red_spiders'_eggs", 'Ruby_necklace', 'Amulet_of_strength', "Green_d'hide_vamb", 'Staff_of_fire', \
	# 	'Blue_wizard_robe', 'Adamant_axe', 'Adamant_scimitar', 'Zamorak_monk_top', 'Staff_of_water', 'Staff_of_air', \
	# 		'Adamantite_bar', 'Amulet_of_power', "Green_d'hide_chaps", 'Mithril_platebody', 'Zamorak_monk_bottom', \
	# 			"Green_d'hide_body", 'Rune_axe', 'Adamant_platebody', 'Runite_ore', 'Rune_scimitar', 'Rune_pickaxe', \
	# 				'Rune_full_helm', 'Rune_kiteshield', 'Rune_2h_sword', 'Rune_platelegs', 'Rune_platebody', 'Old_school_bond']
	
	items_predicted = ['Amulet_of_strength', "Green_d'hide_vamb", 'Staff_of_fire', 'Zamorak_monk_top', 'Staff_of_air', \
		'Adamantite_bar', 'Zamorak_monk_bottom', 'Adamant_platebody', 'Runite_ore', 'Rune_scimitar', 'Rune_pickaxe', \
			'Rune_full_helm', 'Rune_kiteshield', 'Rune_2h_sword', 'Rune_platelegs', 'Rune_platebody', 'Old_school_bond']
				
	data = {}
	names = {}
	count = 0 
	
	buy_avg = pd.read_csv('data/rsbuddy/buy_average.csv')
	buy_avg = buy_avg.set_index('timestamp')
	buy_avg = buy_avg.drop_duplicates()
	buy_avg = buy_avg.reset_index()
	buy_avg = buy_avg.replace(to_replace=0, method='ffill')

	for item_predicted in items_predicted:
		df = pd.read_csv('data/predictions/{}.csv'.format(item_predicted))
		# print(item_predicted)
		# print(df.tail(10))

		current_df = buy_avg[['timestamp', item_predicted]]
		current_df = current_df.rename(columns={'timestamp': 'ts', item_predicted: 'real'})

		merged_df = pd.merge_asof(df, current_df, left_on='timestamp', right_on='ts', direction='backward')
		merged_df = merged_df.tail(48)  # Only show the last 48 time steps (24 hours worth of data)
		chart_data = merged_df.to_dict(orient='records')
		data['{}'.format(count)] = chart_data
		names[count] = item_predicted
		count += 1
		# print(data)

	return render_template("index.html", data=data, names=names)

# Webserver's suggestion engine
@app.route('/suggest')
def suggest():
	
	items_predicted = ['Amulet_of_strength', "Green_d'hide_vamb", 'Staff_of_fire', 'Zamorak_monk_top', 'Staff_of_air', \
		'Adamantite_bar', 'Zamorak_monk_bottom', 'Adamant_platebody', 'Runite_ore', 'Rune_scimitar', 'Rune_pickaxe', \
			'Rune_full_helm', 'Rune_kiteshield', 'Rune_2h_sword', 'Rune_platelegs', 'Rune_platebody', 'Old_school_bond']
				
	data = {}
	
	buy_avg = pd.read_csv('data/rsbuddy/buy_average.csv')
	buy_avg = buy_avg.set_index('timestamp')
	buy_avg = buy_avg.drop_duplicates()
	buy_avg = buy_avg.replace(to_replace=0, method='ffill')

	# Get latest real price values 
	temp_last_row = None
	with open('data/predictions/{}.csv'.format(items_predicted[0]), mode='r') as infile:
		for row in reversed(list(csv.reader(infile))):
			temp_last_row = row
			break

	buy_avg = buy_avg.loc[~buy_avg.index.duplicated(keep='first')]
	closest_real_values = buy_avg.iloc[buy_avg.index.get_loc(int(temp_last_row[0]), method='nearest')]

	# Get all values predicted
	for item_predicted in items_predicted:
		with open('data/predictions/{}.csv'.format(item_predicted), mode='r') as infile:
			last_row = None
			for row in reversed(list(csv.reader(infile))):
				last_row = row
				break

			# Currently save only the value predicted by univariate model
			data[item_predicted] = [int(closest_real_values[item_predicted]), int(last_row[1]), int(last_row[1])-int(closest_real_values[item_predicted])]

	return render_template("suggest.html", data=data)

# A route to return all of the available entries in our catalog.
@app.route('/api', methods=['GET'])
def api_all():
	if 'name' in request.args:
	    name = str(request.args['name'])
	else:
	    return "Error: No name field provided. Please specify an name."

	try:
		with open('data/predictions/{}.csv'.format(name), mode='r') as infile:
			
			reader = csv.reader(infile)
			header_row = next(reader)
			print(header_row)
			last_row = []
			for row in reader:
				last_row = row
			mydict = {name:last_row[idx] for idx, name in enumerate(header_row)}
	except EnvironmentError:
		return "File not found. Please specify different name"	

	return jsonify(mydict)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=80)