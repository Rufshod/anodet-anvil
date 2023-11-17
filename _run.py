import json
import os
import time
import sys
from typing import Optional, Union, List
import anvil.server
from flask import Flask, request, send_from_directory, Response

# from multicamcomposepro.camera import CameraManager
# from multicamcomposepro.utils import Warehouse

# Get mccp by path
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, "..", "mccp/src")
sys.path.append(target_dir)

from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import Warehouse

from ano import predict, get_dataloader, model_fit, get_model

# =============== Anvil & Flask =============== #

uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
anvil.server.connect(uplink_key)

app = Flask(__name__)
object_name = "preview"
path_to_images = os.path.join(
    os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good"
)

# ============================================ #

distributions_path = "data_warehouse/distributions"
dataset_path = "data_warehouse/dataset"
camera_config = "camera_config.json"
model_config = "model_config.json"  # TODO!!


@anvil.server.callable
def train_model(
    object_name: str = object_name,
    camera_config: str = camera_config,
    resnet_config: Optional[str] = None,
) -> None:
    """
    Create and fit a model. Stores model parameters in warehouse/distributions/{object_name}.

    Args:
        object_name (str): Name of the object for training.
        camera_config (str): Path to the camera configuration file.
        resnet_config (str, optional): Configuration for the ResNet model.

    Returns:
        None
    """

    # TODO  load parameters (some of them) from model_config

    #  Load angles from json
    with open(camera_config, "r") as file:
        camera_config_file = file.read()
    data = json.loads(camera_config_file)
    angles = [item["Angle"] for item in data]

    if "Skip" in angles:
        angles.remove("Skip")

    #  Should load json for resnet_config here TODO
    model = get_model(
        selected_model="resnet18"
    )  #  TODO resnet_config json file for input arg

    for angle in angles:
        dataloader = get_dataloader(dataset_path, angle, object_name)
        model_fit(model, dataloader, distributions_path, angle, object_name)
        #  This for loop works for now, but a future implementation should be to rewrite the dataloader


# Function to set the object name from Anvil
@anvil.server.callable
def set_object_name(object_input_name: str = "object") -> str:
    """
    Set the global object name.

    Args:
        object_input_name (str): The input name for the object.

    Returns:
        str: The updated object name.
    """
    global object_name
    object_name = object_input_name
    print("Object name set to:", object_name)
    return object_name


# Called when URL is loaded TRAIN IMAGE
@app.route("/<angle>/<image>")
def get_image(angle: str, image: str) -> Union[str, Response]:
    """
    Retrieve an image from the specified directory.

    Args:
        angle (str): The angle of the camera.
        image (str): The name of the image file.

    Returns:
        Union[str, Response]: The image file or an error message.
    """
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
# TEST IMAGE
@app.route("/test/<angle>/<image>")
def get_test_image(angle: str, image: str) -> Union[str, Response]:
    """
    Retrieve an image from the test directory.

    Args:
        angle (str): The angle of the camera.
        image (str): The name of the image file.

    Returns:
        Union[str, Response]: The image file or an error message.
    """
    directory = os.path.join(
        os.getcwd(), "data_warehouse", "dataset", object_name, "test", "good", angle
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
    
# PLOT
@app.route("/plots/<image>")
def get_plot(image: str) -> Union[str, Response]:
    """
    Retrieve an image from the test directory.

    Args:
        angle (str): The angle of the camera.
        image (str): The name of the image file.

    Returns:
        Union[str, Response]: The image file or an error message.
    """
    directory = os.path.join(
        os.getcwd(), "data_warehouse", "plots"
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

# use utils clean folder name function
@anvil.server.callable
def clean_name(name: str) -> str:
    """
    Cleans and returns a folder name using a method from the Warehouse class.

    Args:
        name (str): The name of the folder to be cleaned.

    Returns:
        str: The cleaned folder name.
    """
    warehouse = Warehouse()
    warehouse.clean_folder_name(name)
    return name


@anvil.server.callable
def get_distribution_list(distributions_path: str = distributions_path) -> List[str]:
    """
    Retrieves a list of directory names in the specified distributions path.

    Args:
        distributions_path (str, optional): The path to the distributions directory.
                                            Defaults to the global distributions_path.

    Returns:
        List[str]: A list of directory names found in the distributions path.
                   Returns a message if no distributions are saved.
    """
    folder_contents = os.listdir(distributions_path)
    return folder_contents if folder_contents else "No distributions saved!"

@anvil.server.callable
def get_angles_from_warehouse(object_name):
    # Quick fix, should be properly stored and accessed from json camera config TODO
    angles = os.listdir(f"data_warehouse/dataset/{object_name}/train/good")
    return angles



@anvil.server.callable
def run_prediction(object_name: str, cam_name: str, distributions_path: str = distributions_path) -> None:
    """
    Runs a prediction on a specified object using predefined camera settings and test images.

    Args:
        object_name (str): The name of the object to run the prediction on.
        distributions_path (str, optional): The path to the model distributions.
                                            Defaults to the global distributions_path.

    Returns:
        None
    """
    print(object_name, cam_name, distributions_path)

    # TODO update hardcoded cam_names with input strings from Anvil
    image_classifications, image_scores, score_maps = predict(
        distributions_path,
        cam_name=cam_name,
        object_name=object_name,

        # test_images: replace with image captured in Anvil
        test_images=[
            f"data_warehouse/dataset/{object_name}/test/good/{cam_name}/000.png"
        ],


        THRESH=13,
    )

    # Cannot return tensors to Anvil
    # Do we want to return any of image_classifications, image_scores, score_maps (?) if so - they need to be converted to ndarray or list or smt
    return

@anvil.server.callable
def get_image_url(angle, image_name):
    """Returns the URL of the image on the Flask server so that it can be displayed in the Anvil app"""
    # Use the current time as a cache-busting query parameter
    print("GETTING IMAGE GET IMAGE URL")
    timestamp = int(time.time())
    image_path = f"http://127.0.0.1:5000/{angle}/{image_name}?cb={timestamp}"
    return image_path

@anvil.server.callable
def get_test_image_url(angle, image_name):
    """Returns the URL of the image from the test directory on the Flask server so that it can be displayed in the Anvil app"""
    # Use the current time as a cache-busting query parameter
    timestamp = int(time.time())
    # Update the URL to point to the new Flask route for test images
    image_path = f"http://127.0.0.1:5000/test/{angle}/{image_name}?cb={timestamp}"
    return image_path

@anvil.server.callable
def get_plot_url():
    """Returns the URL of the image from the test directory on the Flask server so that it can be displayed in the Anvil app"""
    # Use the current time as a cache-busting query parameter
    "getting plot url"
    timestamp = int(time.time())
    # Update the URL to point to the new Flask route for test images
    image_path = f"http://127.0.0.1:5000/plots/plot_0.png?cb={timestamp}"
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
def capture_initial_images() -> None:
    """
    Captures initial images for each camera specified in the camera_config.json file.

    Returns:
        None
    """
    path_to_config = "camera_config.json"

    if os.path.exists(path_to_config) and os.path.getsize(path_to_config):
        print("Capturing initial images")
        warehouse = Warehouse()
        warehouse.build("preview", [])
        camera_manager = CameraManager(
            warehouse, train_images=1, test_anomaly_images=0, allow_user_input=False
        )
        camera_manager.run()
        print("Done capturing initial images")
    else:
        print("camera_config.json does not exist or is empty.")


@anvil.server.callable
def capture_image(object_input_name: str = "object") -> None:
    """
    Captures a single image from the camera for the specified object.

    Args:
        object_input_name (str, optional): The name of the object to capture the image for. Defaults to "object".

    Returns:
        None
    """
    print("Capturing image")
    warehouse = Warehouse()
    warehouse.build(object_name=object_input_name, anomalies=[])

    camera_manager = CameraManager(
        warehouse,
        train_images=1,
        test_anomaly_images=0,
        allow_user_input=False,
        overwrite_original=False,
    )
    camera_manager.run()
    print(warehouse)

@anvil.server.callable
def capture_test_image(object_input_name: str = "object") -> None:
    """
    Captures a single image from the camera for the specified object.

    Args:
        object_input_name (str, optional): The name of the object to capture the image for. Defaults to "object".

    Returns:
        None
    """
    print("Capturing image")
    warehouse = Warehouse()
    warehouse.build(object_name=object_input_name, anomalies=[])

    camera_manager = CameraManager(
        warehouse,
        train_images=0,
        test_anomaly_images=1,
        allow_user_input=False,
        overwrite_original=True,
    )
    camera_manager.run()
    print(warehouse)



# Just some presets values
@anvil.server.callable
def get_preset(key):
    presets = {
        "backbone": ["ResNet18", "Wide_ResNet50"],
        "resize": [112, 224, 448, 896],
        "activation": ["ReLU"],
        "batch_size": [8, 16, 32],
        "device": "cpu",
        "gaussian_blur": True,
    }

    if key == "resize":
        return [str(value) for value in presets.get(key, [])]
    else:
        return presets.get(key, [])

    return presets.get(key)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    anvil.server.wait_forever()
