import json
import anvil.server

uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key"]
# Connect to Anvil server
anvil.server.connect(uplink_key)
print("Connected to Anvil server")


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
    # Take picture
    # Maybe update something in fronend

if __name__ == "__main__":
    anvil.server.wait_forever()
