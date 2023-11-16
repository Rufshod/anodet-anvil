import os
import sys
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Set the backend to non-interactive one (Agg)
import matplotlib.pyplot as plt

import cv2
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Dict, Any, Union, List, Tuple, Any

current_dir = os.getcwd()
target_dir = os.path.join(current_dir, "..", "anodet")
sys.path.append(target_dir)

from anodet import Padim, AnodetDataset, to_batch, classification, visualization


# TODO move to json
config_resnet = {
    "backbone": "ResNet18",
    "preprocessing": {
        "resize": 224,
        "normalize": {"mean": [1.485, 1.456, 1.406], "std": [1.229, 1.224, 1.225]},
    },
    "activation": "ReLU",
    "batch_size": 32,
    "device": "cpu",
    "gaussian_blur": True,
    "threshold": 13,
}

# Backbone resnet18 or wide_resnet50 TODO add support for other resnet
# TODO add layer_hooks
# TODO set parameters for ResnetEmbeddingsExtractor


# Should we set global paths and object name?


def get_model(selected_model: str = "resnet18") -> Padim:
    """
    Get the specified model.

    Args:
        selected_model (str): The name of the model to retrieve.

    Returns:
        Padim: The requested model.
    """

    if selected_model == "resnet18":
        model = Padim("resnet18")
    elif selected_model == "wide_resnet50":
        pass
    elif selected_model == "patchcore":
        pass

    return model


# Run images in dataloader(train) and test images through this pipeline
def build_preprocessing(preprocessing_config: Dict[str, Any]) -> transforms.Compose:
    """
    Build the preprocessing pipeline.

    Args:
        preprocessing_config (Dict[str, Any]): Configuration for preprocessing.

    Returns:
        transforms.Compose: The preprocessing pipeline.
    """

    print("Preprocessing Config:", preprocessing_config)

    resize_size = preprocessing_config["preprocessing"]["resize"]
    normalize = transforms.Normalize(
        mean=preprocessing_config["preprocessing"]["normalize"]["mean"],
        std=preprocessing_config["preprocessing"]["normalize"]["std"],
    )
    preprocessing_pipeline = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return preprocessing_pipeline


# Used during train to batch and pre-process images
def get_dataloader(dataset_path: str, cam_name: str, object_name: str) -> DataLoader:
    """
    Get a DataLoader for the specified dataset.

    Args:
        dataset_path (str): Path to the dataset.
        cam_name (str): Name of the camera.
        object_name (str): Name of the object.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    dataset = AnodetDataset(
        image_directory_path=os.path.join(
            dataset_path, f"{object_name}/train/good/{cam_name}"
        ),
        mask_directory_path=None,
        image_transforms=build_preprocessing(
            config_resnet
        ),  # Maybe should have default values
        mask_transforms=None,
    )
    batch_size = config_resnet.get(
        "batch_size", 32
    )  # Defaults to 32 if not specified in the config
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print(
        f"DataLoader created with {len(dataset)} images for {object_name}, camera {cam_name}."
    )
    return dataloader


def model_fit(
    model: Any,
    dataloader: DataLoader,
    distributions_path: str,
    cam_name: str,
    object_name: str,
) -> None:
    """
    Fits the model using the provided dataloader and saves the model parameters.

    Args:
        model (Any): The model to be fitted.
        dataloader (DataLoader): The DataLoader providing the data for training.
        distributions_path (str): Path to save the model distributions.
        cam_name (str): Name of the camera.
        object_name (str): Name of the object.

    Returns:
        None
    """
    model.fit(dataloader)

    # Create object dir
    save_path = os.path.join(distributions_path, object_name)
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(
        distributions_path, f"{object_name}/{object_name}_{cam_name}"
    )
    torch.save(model.mean, save_path + "_mean.pt")
    torch.save(model.mean, save_path + "_cov_inv.pt")
    print(f"Parameters saved at {distributions_path}")


# Semi-implemented in _run.py
def predict(
    distributions_path: str,
    cam_name: str,
    object_name: str,
    test_images: List[str],
    THRESH: int = config_resnet.get("threshold", 13),
) -> Tuple[Any, Any, Any]:
    """
    Predicts anomalies in the provided images using the trained model.

    Args:
        distributions_path (str): Path to the model distributions.
        cam_name (str): Name of the camera.
        object_name (str): Name of the object.
        test_images (List[str]): List of paths to the test images.
        THRESH (int, optional): Threshold value for anomaly detection. Defaults to config_resnet's threshold.

    Returns:
        Tuple[Any, Any, Any]: Returns a tuple containing image classifications, image scores, and score maps.
    """
    # While this IS built for multiple images, it's not built for multiple angles.
    # Meaning we have to predict on all unique angles as unique instances.

    # An idea - perhaps - is to modify the DataLoader / AnodetDataset to hold images for all angles

    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in test_images]
    device = config_resnet.get("device", "cpu")
    distributions_path = os.path.join(
        distributions_path, f"{object_name}/{object_name}_{cam_name}"
    )

    # Preprocessing test image(s)
    batch = to_batch(images, build_preprocessing(config_resnet), torch.device(device))

    # Load model (mean and cov_inv)
    mean = torch.load(distributions_path + "_mean.pt")
    cov_inv = torch.load(distributions_path + "_cov_inv.pt")
    padim = Padim(
        backbone="resnet18", mean=mean, cov_inv=cov_inv, device=torch.device(device)
    )

    # Predict and classify
    image_scores, score_maps = padim.predict(
        batch, gaussian_blur=config_resnet.get("gaussian_blur")
    )
    score_map_classifications = classification(score_maps, THRESH)
    image_classifications = classification(image_scores, THRESH)

    # Plots
    test_images = np.array(images).copy()

    boundary_images = visualization.framed_boundary_images(
        test_images, score_map_classifications, image_classifications, padding=40
    )
    heatmap_images = visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = visualization.highlighted_images(
        images, score_map_classifications, color=(128, 0, 128)
    )

    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle(f"Image: {idx}", y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])

        plt.savefig(
            f"data_warehouse/plots/plot_{idx}.png"
        )  # TODO save to warehouse/plots/object_name/angle
        plt.close()

    print("Saved figure at data_warehouse/plots.")

    return image_classifications, image_scores, score_maps


if __name__ == "__main__":
    # Should be the same for the entire app
    dataset_path = "data_warehouse/dataset"
    distributions_path = "data_warehouse/distributions"

    # Should be set for user
    cam_name = "Front"
    object_name = ["test_object"]

    # Train
    if True:
        dataloader = get_dataloader(dataset_path, cam_name, object_name[0])
        backbone_name = config_resnet["backbone"]
        model = Padim(backbone=backbone_name.lower())
        model_fit(model, dataloader, distributions_path, cam_name, object_name[0])

    predict(
        distributions_path="data_warehouse/distributions",
        cam_name="Front",
        object_name="test_object",
        test_images=["data_warehouse/dataset/test_object/test/good/Front/001.png"]
,
    )
