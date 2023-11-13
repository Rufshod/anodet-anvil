import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive one (Agg)
import matplotlib.pyplot as plt
import sys

# Get anodet by path
sys.path.append('/Users/helvetica/_master_anodet/anodet')
from anodet import Padim, AnodetDataset, to_batch, classification, visualization, standard_image_transform

def get_dataloader(dataset_path, cam_name, object_name):
    dataset = AnodetDataset(os.path.join(dataset_path, f"{object_name}/train/good/{cam_name}"))
    dataloader = DataLoader(dataset, batch_size=32)
    print(f"DataLoader created with {len(dataset)} images for {object_name}, camera {cam_name}.")
    return dataloader

def model_fit(model, dataloader, distributions_path, cam_name, object_name):
    model.fit(dataloader)
    torch.save(model.mean, os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_mean.pt"))
    torch.save(model.cov_inv, os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_cov_inv.pt"))
    print(f"Parameters saved at {distributions_path}")

def predict(distributions_path, cam_name, object_name, test_images, THRESH=13):
    print("ano - predict running")

    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in test_images]
    batch = to_batch(images, standard_image_transform, torch.device("cpu"))

    mean = torch.load(os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_mean.pt"))
    cov_inv = torch.load(os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_cov_inv.pt"))

    padim = Padim(backbone="resnet18", mean=mean, cov_inv=cov_inv, device=torch.device("cpu"))
    image_scores, score_maps = padim.predict(batch)

    score_map_classifications = classification(score_maps, THRESH)
    image_classifications = classification(image_scores, THRESH)

    test_images = np.array(images).copy()

    boundary_images = visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
    heatmap_images = visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle(f"Image: {idx}", y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        
        plt.savefig(f"data_warehouse/plots/plot_{idx}.png")
        plt.close()

    print("Saved figures.")
        
    return image_classifications, image_scores, score_maps
