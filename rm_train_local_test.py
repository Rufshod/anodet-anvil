# CAN BE REMOVED!!

import json
from ano import get_dataloader, model_fit, get_model

dataset_path = "data_warehouse/dataset"
distributions_path = "data_warehouse/distributions"


object_name = "november"
camera_config = "camera_config.json"


def train_model(
    object_name=object_name, camera_config=camera_config, resnet_config=None
):
    #  Load angles from json (Should this json store object name as well(?))
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


train_model()
