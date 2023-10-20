import anvil.server
import anvil.users
import anvil.tables as tables
from anvil.tables import app_tables
import json
uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]

# get the Server_Uplink_Key from anvil_key.json
anvil.server.connect(uplink_key)

app_tables.testdb.add_row(name="test")


# print the last row in the table
print(app_tables.testdb.search())

# remove all rows from the table
#for row in app_tables.testdb.search():
    #row.delete()

# disconnect from the server

anvil.server.disconnect()

