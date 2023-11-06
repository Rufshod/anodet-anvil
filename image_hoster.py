from flask import Flask, send_from_directory
import os

app = Flask(__name__)
angle = "Left"
object_name = "default2_object"
path_to_images = os.path.join(os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good", angle)
print(path_to_images)
@app.route('/<angle>/<image>')
def get_image(angle, image):
    directory = os.path.join(os.getcwd(), "data_warehouse", "dataset", object_name, "train", "good", angle)
    return send_from_directory(directory, image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
