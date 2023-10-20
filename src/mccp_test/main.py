import anvil.server  # Anvil server import
from .camera import CameraManager
from .utils import CameraConfigurator, CameraIdentifier, Warehouse
# anvil key
import json
uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
# Connect to Anvil server
anvil.server.connect(uplink_key)

object_name = "anvil_test"


def main():
    # Existing code
    camera_identifier = CameraIdentifier()
    camera_identifier.camera_identifier()  # Check if camera_config.json exists

    camera_configurator = CameraConfigurator()
    camera_configurator.camera_configurator()

    warehouse = Warehouse()
    warehouse.build(
        object_name=object_name,
        anomalies=["db_anomaly"],
    )
    print(warehouse)

    camera_manager = CameraManager(warehouse, 2, 3)
    camera_manager.run()


if __name__ == "__main__":
    main()
