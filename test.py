import requests
import json

itemID = 560
r = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/catalogue/detail.json', params={'item': itemID})
json_data = json.loads(r.text)
json_formatted_str = json.dumps(json_data, indent=2)

print(json_formatted_str)