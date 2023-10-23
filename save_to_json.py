import json

import anvil.server

uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
# Connect to Anvil server
anvil.server.connect(uplink_key)
print("Connected to Anvil server")


@anvil.server.callable
def save_to_json(num):
    data = {"number": num}

    with open("number.json", "w") as file:
        json.dump(data, file)


@anvil.server.callable
def load_from_json():
    with open("number.json", "r") as file:
        data = json.load(file)
        return data["number"]


anvil.server.wait_forever()
