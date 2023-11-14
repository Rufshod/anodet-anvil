import json
import os
import time
import sys

import anvil.server
from flask import Flask, request, send_from_directory
# from multicamcomposepro.camera import CameraManager
# from multicamcomposepro.utils import Warehouse

# Get mccp by path
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, '..', 'mccp/src')
sys.path.append(target_dir)

from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import Warehouse

from ano import predict


# Connect to Anvil server
uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
anvil.server.connect(uplink_key)
print("Connected to Anvil server")

## Flask server to serve images to Anvil
app = Flask(__name__)
object_name = "preview"
path_to_images = os.path.join(os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good")
distributions_path = os.path.join(os.getcwd(), "data_warehouse", "distributions")


# Called when URL is loaded
@app.route("/<angle>/<image>")
def get_image(angle, image):
    directory = os.path.join(
        os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good", angle
    )
    # Add a cache-busting query parameter
    cache_buster = request.args.get("cb", int(time.time()))

    if os.path.isfile(os.path.join(directory, image)):
        response = send_from_directory(directory, image)
        
        # Modify the cache control headers
        response.headers["Cache-Control"] = "no-store"
        return response
    else:
        return "File not found", 404

@anvil.server.callable
def get_distribution_list(distributions_path=distributions_path):
    # List the DIR NAMES in data_warehouse/distributions
    folder_contents = os.listdir(distributions_path)
    return folder_contents if folder_contents else "No distributions saved!"

@anvil.server.callable
def run_prediction(object_name, distributions_path=distributions_path):

    # TODO update hardcoded cam_names with input strings from Anvil
    
    # for angle in angles:
    predict(distributions_path, cam_name="cam_0_left", object_name=object_name, test_images=["/Users/helvetica/anodet-anvil/data_warehouse/dataset/purple_duck/test/albinism/cam_0_left/023.png"], THRESH=13)



@anvil.server.callable
def get_image_url(angle, image_name):
    """Returns the URL of the image on the Flask server so that it can be displayed in the Anvil app"""
    # Use the current time as a cache-busting query parameter
    timestamp = int(time.time())
    image_path = f"http://127.0.0.1:5000/{angle}/{image_name}?cb={timestamp}"
    return image_path


@anvil.server.callable
def save_to_json(data):
    """Saves the camera data to a JSON file"""
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
    """Loads the camera data from a JSON file"""
    try:
        with open("camera_config.json", "r") as file:
            data = json.load(file)
            return data
    except (json.JSONDecodeError, FileNotFoundError):  # Handle empty or missing file
        print("camera_config.json not found")
        return []


@anvil.server.callable
def capture_initial_images():
    """Captures the initial images for the cameras in the camera_config.json file"""
    path_to_config = "camera_config.json"

    if os.path.exists(path_to_config) and os.path.getsize(path_to_config) > 0:
        print("Capturing initial images")
        warehouse = Warehouse()
        warehouse.build("preview", [])
        camera_manager = CameraManager(
            warehouse, train_images=1, test_anomaly_images=0, allow_user_input=False
        )
        camera_manager.run()
        return print("Done capturing initial images")
    else:
        print("camera_config.json does not exist or is empty.")


@anvil.server.callable
def run_mccp():
    """Runs the multicam compose pro program"""
    print("mccp running")
    # Get settings from json
    warehouse = Warehouse()
    warehouse.build()

    # Take picture
    camera_manager = CameraManager(warehouse, train_images=1, test_anomaly_images=0, allow_user_input=False)
    camera_manager.run()
    print(warehouse)

@anvil.server.callabale
def capture_train_images():
    
    pass

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    anvil.server.wait_forever()

