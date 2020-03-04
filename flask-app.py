from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
import json
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route('/')
def index():
	# items_predicted = ['Old_school_bond']
	items_predicted = ['Old_school_bond', 'Rune_platebody', 'Adamant_platebody']
	data = {}
	names = {}
	count = 0 
	for item_predicted in items_predicted:
		df = pd.read_csv('data/predictions/{}.csv'.format(item_predicted))
		# print(df.tail(10))

		buy_avg = pd.read_csv('data/rsbuddy/buy_average.csv')[['timestamp', item_predicted]]
		buy_avg = buy_avg.set_index('timestamp')
		buy_avg = buy_avg.drop_duplicates()
		buy_avg = buy_avg.reset_index()
		buy_avg = buy_avg.rename(columns={'timestamp': 'ts', item_predicted: 'real'})
		buy_avg = buy_avg.replace(to_replace=0, method='ffill')
		# print(buy_avg.tail(10))

		merged_df = pd.merge_asof(df, buy_avg, left_on='timestamp', right_on='ts', direction='backward')
		chart_data = merged_df.to_dict(orient='records')
		chart_data = json.dumps(chart_data, indent=2)
		data['chart{}'.format(count)] = chart_data
		names[count] = item_predicted
		count += 1
		# print(data)

	return render_template("index.html", data=data, names=names)

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
			next(reader)
			mydict = {rows[0]:rows[1] for rows in reader}
	except EnvironmentError:
		return "File not found. Please specify different name"	

	return jsonify(mydict)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=80)