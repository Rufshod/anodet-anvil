from flask import Flask, send_from_directory
import os
import anvil.server
import json


object_name = "default2_object"
uplink_key = json.load(open("anvil_key.json"))["Server_Uplink_Key_for_flask"]
# Connect to Anvil server
anvil.server.connect(uplink_key)
print("Connected to Anvil server")


@anvil.server.callable
def get_image_url(angle, image_name):
    image_path = f'http://127.0.0.1:5000/{angle}/{image_name}'
    return image_path

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
