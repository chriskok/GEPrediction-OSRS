import requests
import json
import csv

csvName = 'Rune_Data'

# get more item codes here: https://everythingrs.com/tools/osrs/itemlist/238
# itemList = [1521, 1519, 1517, 1515]  # Logs
itemList = list(range(554,567,1))  # Runes

fullDict = {}
labels = ['timestamp']

# Construct dictionary full of data
for itemID in itemList:
    r = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/graph/{}.json'.format(itemID))
    json_data = json.loads(r.text)
    current_daily_dict = json_data['daily']

    for daily_timestamp in current_daily_dict:
        if (daily_timestamp in fullDict):
            fullDict[daily_timestamp].append(current_daily_dict[daily_timestamp])
        else:
            fullDict[daily_timestamp] = [current_daily_dict[daily_timestamp]]
    
    r2 = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/catalogue/detail.json', params={'item': itemID})
    labels.append(json.loads(r2.text)['item']['name'].replace(" ", "_"))

# print(fullDict)


# Write to CSV file
with open('data/{}.csv'.format(csvName), mode='w', newline='') as GE_data:
    GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    GE_writer.writerow(labels)  # write field names

    for daily_timestamp in fullDict:
        new_array = [daily_timestamp]
        new_array.extend(fullDict[daily_timestamp])
        # print(new_array)
        GE_writer.writerow(new_array)