import json

import anvil.server
from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import Warehouse

uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
# Connect to Anvil server
anvil.server.connect(uplink_key)
print("Connected to Anvil server")


## Flask server to upload images

from flask import Flask, send_from_directory
import os

app = Flask(__name__)
object_name = "default2_object"
path_to_images = os.path.join(os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good")
print(path_to_images)
@app.route('/<angle>/<image>')
def get_image(angle, image):
    directory = os.path.join(os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good", angle)
    print("Directory:", directory)
    print("Image:", image)
    if os.path.isfile(os.path.join(directory, image)):
        return send_from_directory(directory, image)
    else:
        return "File not found", 404


@anvil.server.callable
def get_image_url(angle, image_name):
    image_path = f'http://127.0.0.1:5000/{angle}/{image_name}'
    return image_path


@anvil.server.callable
def save_to_json(data):
    print("Received data:", data)
    camera_data = data[0]  # Extract the dictionary from the list
    camera_id = camera_data["Camera"]
    existing_data = load_from_json()

    # Check if camera with the same ID exists
    existing_camera = next(
        (camera for camera in existing_data if camera["Camera"] == camera_id), None
    )
    if existing_camera:
        # Update the existing camera data
        existing_camera.update(camera_data)
    else:
        # Add the new camera data to the list
        existing_data.append(camera_data)

    # Save the updated data to the JSON file
    with open("camera_config.json", "w") as file:
        json.dump(existing_data, file, indent=4)


@anvil.server.callable
def load_from_json():
    try:
        with open("camera_config.json", "r") as file:
            data = json.load(file)
            return data
    except (json.JSONDecodeError, FileNotFoundError):  # Handle empty or missing file
        print("camera_config.json not found")
        return []


@anvil.server.callable
def run_mccp():
    print("mccp running")
    # Get settings from json
    warehouse = Warehouse()
    warehouse.build()

    # Take picture
    camera_manager = CameraManager(warehouse, train_images=1, test_anomaly_images=0, allow_user_input=False)
    camera_manager.run()
    print(warehouse)

    # Maybe update something in frontend :shrug:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    anvil.server.wait_forever()

