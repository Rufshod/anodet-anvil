import anvil.server
import json
uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]

# get the Server_Uplink_Key from anvil_key.json
anvil.server.connect(uplink_key)
