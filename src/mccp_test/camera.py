import json
import logging
import os
from platform import system
from typing import List

import cv2

from utils import Warehouse

logging.basicConfig(level=logging.INFO)
os_name = system()


class CameraManager:
    def __init__(
        self, warehouse: Warehouse, test_anomaly_images: int = 5, train_images: int = 10
    ) -> None:
        """
        Initialize the CameraManager object.

        This method sets up the initial state of the CameraManager, including loading camera configurations and mappings.

        :param warehouse: Warehouse object containing object and anomaly information. This object guides the image capturing process.
        :param test_anomaly_images: Number of test anomaly images to capture. If set to 5, it means 5 test anomaly images will be captured.
        :param train_images: Number of training images to capture.

        :raises: TODO Add exceptions.

        Example:
            warehouse = Warehouse()
            camera_manager = CameraManager(warehouse, test_anomaly_images=50, train_images=200)
        """
        self.warehouse: Warehouse = warehouse
        self.test_anomaly_images: int = test_anomaly_images
        self.train_images: int = train_images
        self.captures: List = []
        self.load_camera_config()
        self.load_camera_mapping()
        self.camera_angles: List = [
            "cam_0_left",
            "cam_1_right",
            "cam_2_front",
            "cam_3_front_left",
            "cam_4_front_right",
            "cam_5_back",
            "cam_6_back_left",
            "cam_7_back_right",
            "cam_8_top",
            "cam_9_top_left",
            "cam_10_top_right",
        ]
        self.sort_camera_angles()

        # Initialize cameras

    def initialize_cameras(self) -> None:
        """
        Initialize the camera settings and configurations.

        This method sets up the camera configurations based on the operating system
        and pre-defined settings. It should be called before capturing any images.

        :raises: TODO Add exceptions.
        """
        self.captures = []
        for cam_idx in range(self.num_cameras):
            print(f"Camera {cam_idx} initializing...")
            cap = (
                cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if os_name == "Windows"
                else cv2.VideoCapture(cam_idx)
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.color_temp)
            self.captures.append(cap)

    def __del__(self) -> None:
        for cap in self.captures:
            cap.release()

    def load_camera_config(self, filename: str = "camera_config.json") -> None:
        """
        Load camera configurations from a JSON file.

        This method reads camera settings like exposure and color temperature from a JSON file.

        :param filename: Name of the JSON file containing camera configurations.

        :raises FileNotFoundError: If the specified JSON file is not found.

        Example:
            load_camera_config("custom_camera_config.json")
        """
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                camera_settings = data.get("CameraSettings", {})
                self.exposure = camera_settings.get("Camera Exposure", 0)
                self.color_temp = camera_settings.get("Camera Color Temperature", 0)
                self.num_cameras = len(data.get("Camera Order", {}))
        else:
            logging.warning(f"{filename} not found! Using default camera settings.")

    def load_camera_mapping(self, filename: str = "camera_config.json") -> None:
        """ """
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                self.camera_mapping = data.get("Camera Order", {})
        else:
            logging.warning(
                "camera_config.json not found! Please run camera_identifier.py first."
            )

    def sort_camera_angles(self) -> None:
        """
        Sort camera angles based on the loaded camera mapping.

        This method sorts the camera angles based on the camera mapping loaded from the JSON file.

        :raises: TODO Add exceptions.

        Example:
            sort_camera_angles()

        """
        if hasattr(self, "camera_mapping"):
            sorted_angles = [None] * len(self.camera_angles)
            for identifier, index in self.camera_mapping.items():
                if identifier == "skip":
                    if isinstance(index, list):
                        for i in index:
                            sorted_angles[i] = "skip"
                    else:
                        sorted_angles[index] = "skip"
                else:
                    sorted_angles[index] = self.camera_angles[int(identifier)]
            self.camera_angles = sorted_angles
            print("Sorted Camera Angles:", self.camera_angles)  # Debugging line

    def capture_multiple_images(
        self, folder_path: str, num_pictures_to_take: int
    ) -> None:
        """
        Capture multiple images from different cameras and angles.

        This method captures a specified number of images from various camera angles and saves them in the given folder.

        :param folder_path: Directory where the captured images will be saved.
        :param num_pictures_to_take: Number of pictures to capture.

        :raises SomeException: If capturing fails.

        Example:
            capture_multiple_images("/path/to/save", 5)
        """
        if num_pictures_to_take == 0:
            logging.info("Skipping capture due to test_anomaly_images set to 0.")
            return

        image_counter = 0
        for _ in range(num_pictures_to_take):
            # Pause here to allow for object adjustment
            input("Press Enter to continue capturing after adjusting the object...")
            for cam_idx, angle in enumerate(self.camera_angles):
                self.capture_single_image(folder_path, cam_idx, angle, image_counter)
            image_counter += 1

    def capture_training_and_test_images(self) -> None:
        """
        Capture both training and test images for good objects.

        This method captures both training and test images for good objects. It prompts the user to adjust the object before capturing.

        :raises: TODO Add exceptions.

        Example:
            capture_good_object()
        """
        base_dir = os.path.join(
            os.getcwd(), "data_warehouse", "dataset", self.warehouse.object_name
        )

        if self.train_images != 0:
            folder_type = "train"
            input(
                f"Press Enter to capture TRAINING images for {self.warehouse.object_name} in {folder_type}:"
            )
            good_folder = os.path.join(base_dir, folder_type, "good")
            self.capture_multiple_images(good_folder, self.train_images)
            logging.info(f"Captured images for good object in {folder_type} folder.")

        if self.test_anomaly_images != 0:
            folder_type = "test"
            input(
                f"Press Enter to capture images for good object in {folder_type} folder:"
            )
            good_folder = os.path.join(base_dir, folder_type, "good")
            self.capture_multiple_images(good_folder, self.test_anomaly_images)
            logging.info(f"Captured images for good object in {folder_type} folder.")

    def capture_single_image(
        self, folder_path: str, cam_idx: int, angle: str, image_counter: int
    ) -> None:
        """
        Capture a single image from a specific camera angle.

        This method captures a single image from a specific camera angle and saves it in the given folder.
        Its main purpose is to be used within the capture_multiple_images() method but can be called directly if needed.

        :param folder_path: Directory where the captured image will be saved.
        :param cam_idx: Index of the camera to use for capturing.
        :param angle: Camera angle for capturing.
        :param image_counter: Counter for the image to be captured.

        :raises: TODO Add exceptions.

        Example:
            capture_single_image("/path/to/save", 0, "cam_0_left", 0)
        """
        if angle is None or angle == "skip":
            return

        angle_folder_path = os.path.join(folder_path, angle)
        os.makedirs(angle_folder_path, exist_ok=True)

        # Use the pre-initialized capture object
        cap = self.captures[cam_idx]

        # Flush the buffer
        for _ in range(2):
            ret, _ = cap.read()
            if not ret:
                logging.error(
                    f"Could not read frame from camera {cam_idx} at angle {angle}."
                )
                return

        # Capture the actual frame
        ret, frame = cap.read()
        if not ret:
            logging.error(
                f"Could not read frame from camera {cam_idx} at angle {angle}."
            )
            return

        filename = os.path.join(angle_folder_path, f"{image_counter:03d}.png")
        cv2.imwrite(filename, frame)
        logging.info(f"Saved image {filename}")

    def run(self) -> None:
        """
        Main function to run the camera capturing process.

        This method is the main function to run the camera capturing process. It prompts the user to adjust the object before capturing.

        :raises: TODO Add exceptions.

        Example:
            run()
        """
        print(self.warehouse.anomalies)

        self.initialize_cameras()
        self.capture_training_and_test_images()
        base_dir = os.path.join(
            os.getcwd(), "data_warehouse", "dataset", self.warehouse.object_name
        )
        for anomaly in self.warehouse.anomalies:
            input(f"Press Enter to capture images for anomaly: {anomaly}")
            cleaned_anomaly = self.warehouse.clean_folder_name(
                anomaly
            )  # Clean the anomaly name
            anomaly_folder = os.path.join(
                base_dir, "test", cleaned_anomaly
            )  # Use the cleaned name
            self.capture_multiple_images(anomaly_folder, self.test_anomaly_images)
            logging.info(f"Captured images for anomaly: {anomaly}")


if __name__ == "__main__":
    warehouse = Warehouse()
    warehouse.build("train_image_test", ["anomaly_1", "anomaly_2"])
    camera_manager = CameraManager(warehouse, test_anomaly_images=5, train_images=10)
    camera_manager.run()