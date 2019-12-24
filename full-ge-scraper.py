import requests
import json
import csv
import time

rsbuddyAPI = "https://rsbuddy.com/exchange/summary.json"

# One dict each for 'buy_average', 'buy_quantity', 'sell_average', 'sell_quantity', 'overall_average', 'overall_quantity'
buy_average = {}
buy_quantity = {}
sell_average = {}
sell_quantity = {}
overall_average = {}
overall_quantity = {}

labels = ['timestamp']
chosen_ids = []
MINUTES_TO_WAIT = 5

def initialize_data():
    # Get the seconds since epoch
    current_timestamp = int(time.time())
    print("{} - initializing data".format(current_timestamp))
    buy_average[current_timestamp] = []
    buy_quantity[current_timestamp] = []
    sell_average[current_timestamp] = []
    sell_quantity[current_timestamp] = []
    overall_average[current_timestamp] = []
    overall_quantity[current_timestamp] = []

    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)
    for item in json_data:
        if (json_data[item]["overall_quantity"] > 5 and json_data[item]["members"] == False):
            labels.append(json_data[item]["name"].replace(" ", "_"))
            chosen_ids.append(json_data[item]["id"])
            buy_average[current_timestamp].append(json_data[item]["buy_average"])
            buy_quantity[current_timestamp].append(json_data[item]["buy_quantity"])
            sell_average[current_timestamp].append(json_data[item]["sell_average"])
            sell_quantity[current_timestamp].append(json_data[item]["sell_quantity"])
            overall_average[current_timestamp].append(json_data[item]["overall_average"])
            overall_quantity[current_timestamp].append(json_data[item]["overall_quantity"])

def append_data():
    # Get the seconds since epoch
    current_timestamp = int(time.time())
    print("{} - appending data".format(current_timestamp))

    buy_average[current_timestamp] = []
    buy_quantity[current_timestamp] = []
    sell_average[current_timestamp] = []
    sell_quantity[current_timestamp] = []
    overall_average[current_timestamp] = []
    overall_quantity[current_timestamp] = []

    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)
    for item in json_data:
        if (json_data[item]["id"] in chosen_ids):
            buy_average[current_timestamp].append(json_data[item]["buy_average"])
            buy_quantity[current_timestamp].append(json_data[item]["buy_quantity"])
            sell_average[current_timestamp].append(json_data[item]["sell_average"])
            sell_quantity[current_timestamp].append(json_data[item]["sell_quantity"])
            overall_average[current_timestamp].append(json_data[item]["overall_average"])
            overall_quantity[current_timestamp].append(json_data[item]["overall_quantity"])

def writeToCSV(filename, data):
    with open('data/{}.csv'.format(filename), mode='w', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        GE_writer.writerow(labels)  # write field names

        for time in data:
            new_array = [time]
            new_array.extend(data[time])
            # print(len(new_array))
            GE_writer.writerow(new_array)

def main():
    initialize_data()
    time.sleep(MINUTES_TO_WAIT * 60)

    while True:
        append_data()

        writeToCSV("buy_average", buy_average)
        writeToCSV("buy_quantity", buy_quantity)
        writeToCSV("sell_average", sell_average)
        writeToCSV("sell_quantity", sell_quantity)
        writeToCSV("overall_average", overall_average)
        writeToCSV("overall_quantity", overall_quantity)

        time.sleep(MINUTES_TO_WAIT * 60)

if __name__ == "__main__":
    main()