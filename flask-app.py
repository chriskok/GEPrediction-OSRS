from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
import json
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route('/')
def index():
	item_predicted = 'Adamant_platebody'
	df = pd.read_csv('data/predictions/{}.csv'.format(item_predicted))
	print(df.head())

	buy_avg = pd.read_csv('data/rsbuddy/buy_average.csv')[['timestamp', item_predicted]]
	buy_avg = buy_avg.set_index('timestamp')
	buy_avg = buy_avg.drop_duplicates()
	buy_avg = buy_avg.reset_index()
	buy_avg = buy_avg.rename(columns={'timestamp': 'ts', item_predicted: 'real'})
	print(buy_avg.head())

	merged_df = pd.merge_asof(df, buy_avg, left_on='timestamp', right_on='ts', direction='backward')
	chart_data = merged_df.to_dict(orient='records')
	chart_data = json.dumps(chart_data, indent=2)
	data = {'chart_data': chart_data}
	print(data)

	return render_template("index.html", data=data)

# A route to return all of the available entries in our catalog.
@app.route('/api/item/', methods=['GET'])
def api_all():
	with open('data/predictions/Old_school_bond.csv', mode='r') as infile:
		reader = csv.reader(infile)
		next(reader)
		mydict = {rows[0]:rows[1] for rows in reader}
		
	# # Check if an ID was provided as part of the URL.
	# # If ID is provided, assign it to a variable.
	# # If no ID is provided, display an error in the browser.
	# if 'id' in request.args:
	#     id = int(request.args['id'])
	# else:
	#     return "Error: No id field provided. Please specify an id."

	# # Create an empty list for our results
	# results = []

	# # Loop through the data and match results that fit the requested ID.
	# # IDs are unique, but other fields might return many results
	# for book in books:
	#     if book['id'] == id:
	#         results.append(book)

	return jsonify(mydict)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=80)