import json
import os
import time

import anvil.server
from flask import Flask, request, send_from_directory
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
path_to_images = os.path.join(
    os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good"
)


# TODO path_to_distributions here


# Function to set the object name from Anvil
@anvil.server.callable
def set_object_name(object_input_name: str = "object"):
    global object_name
    object_name = object_input_name
    print("Object name set to:", object_name)
    return object_name


# Called when URL is loaded
@app.route("/<angle>/<image>")
def get_image(angle, image):
    print("RUNNING GET_IMAGE")
    directory = os.path.join(
        os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good", angle
    )
    full_path = os.path.join(directory, image)
    print("Full path to image:", full_path)
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
def get_distribution_list():
    # List the DIR NAMES in data_warehouse/distributions
    folder_path = os.path.join(os.getcwd(), "data_warehouse", "distributions")
    folder_contents = os.listdir(folder_path)
    return folder_contents if folder_contents else "No distributions saved!"


@anvil.server.callable
def run_prediction(object_name):
    print("run_prediction")

    # Same as get_distributions_list - should maybe be in Warehouse class
    distributions_path = os.path.join(os.getcwd(), "data_warehouse", "distributions")

    # TODO update hardcoded cam_names with input strings from Anvil
    predict(
        distributions_path,
        cam_name="cam_0_left",
        object_name=object_name,
        test_images=[
            "/Users/helvetica/_master_anodet/anodet/data_warehouse/dataset/purple_duck/test/albinism/cam_0_left/001.png"
        ],
        THRESH=13,
    )


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
def capture_image(object_input_name: str = "object"):
    """Function to capture a single image from the camera"""
    print("Capturing image")
    # Get settings from json
    warehouse = Warehouse()
    warehouse.build(object_name=object_input_name, anomalies=[])

    # Take picture
    camera_manager = CameraManager(
        warehouse,
        train_images=1,
        test_anomaly_images=0,
        allow_user_input=False,
        overwrite_original=False,
    )
    camera_manager.run()
    print(warehouse)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    anvil.server.wait_forever()
