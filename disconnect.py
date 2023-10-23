import json
from contextlib import contextmanager

import anvil.server

uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]


@contextmanager
def open_anvil_connection(uplink_key):
    anvil.server.connect(uplink_key)
    yield
    anvil.server.disconnect()
