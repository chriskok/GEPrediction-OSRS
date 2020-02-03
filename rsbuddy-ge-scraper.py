import requests
import json
import csv
import time
import os.path

rsbuddyAPI = "https://rsbuddy.com/exchange/summary.json"

# One dict each for 'buy_average', 'buy_quantity', 'sell_average', 'sell_quantity', 'overall_average', 'overall_quantity'
buy_average = []
buy_quantity = []
sell_average = []
sell_quantity = []
overall_average = []
overall_quantity = []

labels = ['timestamp']

def writeToCSV(filename, data, timestamp):
    with open('data/rsbuddy/{}.csv'.format(filename), mode='w', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        GE_writer.writerow(labels)  # write field names

        new_array = [timestamp]
        new_array.extend(data)
        GE_writer.writerow(new_array)


def appendToCSV(filename, data, timestamp):
    with open('data/rsbuddy/{}.csv'.format(filename), mode='a', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        new_array = [timestamp]
        new_array.extend(data)
        GE_writer.writerow(new_array)

def initialize_data():
    # Get the seconds since epoch
    current_timestamp = int(time.time())
    print("{} - initializing data".format(current_timestamp))

    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)
    for item in json_data:
        labels.append(json_data[item]["name"].replace(" ", "_"))
        buy_average.append(json_data[item]["buy_average"])
        buy_quantity.append(json_data[item]["buy_quantity"])
        sell_average.append(json_data[item]["sell_average"])
        sell_quantity.append(json_data[item]["sell_quantity"])
        overall_average.append(json_data[item]["overall_average"])
        overall_quantity.append(json_data[item]["overall_quantity"])
    
    writeToCSV("buy_average", buy_average, current_timestamp)
    writeToCSV("buy_quantity", buy_quantity, current_timestamp)
    writeToCSV("sell_average", sell_average, current_timestamp)
    writeToCSV("sell_quantity", sell_quantity, current_timestamp)
    writeToCSV("overall_average", overall_average, current_timestamp)
    writeToCSV("overall_quantity", overall_quantity, current_timestamp)

def append_data():
    # Get the seconds since epoch
    current_timestamp = int(time.time())
    print("{} - appending data".format(current_timestamp))

    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)
    for item in json_data:
        buy_average.append(json_data[item]["buy_average"])
        buy_quantity.append(json_data[item]["buy_quantity"])
        sell_average.append(json_data[item]["sell_average"])
        sell_quantity.append(json_data[item]["sell_quantity"])
        overall_average.append(json_data[item]["overall_average"])
        overall_quantity.append(json_data[item]["overall_quantity"])

    appendToCSV("buy_average", buy_average, current_timestamp)
    appendToCSV("buy_quantity", buy_quantity, current_timestamp)
    appendToCSV("sell_average", sell_average, current_timestamp)
    appendToCSV("sell_quantity", sell_quantity, current_timestamp)
    appendToCSV("overall_average", overall_average, current_timestamp)
    appendToCSV("overall_quantity", overall_quantity, current_timestamp)


def main():

    if os.path.isfile('data/rsbuddy/buy_average.csv'):
        append_data()
    else:
        initialize_data()

if __name__ == "__main__":
    main()