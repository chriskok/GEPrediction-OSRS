import requests
import json
import time

rsbuddyAPI = "https://rsbuddy.com/exchange/summary.json"

fullDict = {}
labels = ['timestamp']
allitems = []

def initialize_fullDict():
    # Get the seconds since epoch
    current_timestamp = int(time.time())
    fullDict[current_timestamp] = []

    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)
    for item in json_data:
        if (json_data[item]["overall_quantity"] > 5 and json_data[item]["members"] == False):
            print(json_data[item])
            allitems.append(item)

    print(len(allitems))

initialize_fullDict()