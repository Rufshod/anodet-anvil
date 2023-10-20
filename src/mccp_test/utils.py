
from anvil.tables import app_tables
import json
import os
from platform import system
from threading import Event, Thread
from typing import List, Optional, Union

import cv2
import numpy as np

# Camera Text Overlay #


def camera_text_overlay(frame, camera_name):
    """
    Add camera name as text overlay to frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, frame.shape[0] - 10)  # Bottom-left corner frame for each cam stream
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(frame, camera_name, position, font, font_scale, font_color, line_type)


#   Camera Identifier #


class CameraIdentifier:
    """
    Opens Cameras one by one and asks the user to identify them.
    Saves camera order with unique identifiers to camera_config.json.
    Order is used in CameraManager to display the camera streams in the correct order.
    """

    def __init__(self):
        """
        Initialize CameraIdentifier object.

        :ivar camera_mapping: A dictionary containing the camera order with unique identifiers.

        You can change max_tested value to increase/decrease the number of cameras that can be connected to the computer.
        """
        self.camera_mapping: dict = {}
        self.os_name: str = system()
        self.max_tested: int = 10  # Max USB connections / cameras

    def identify_camera(self, index: Union[int, str], event: Event) -> None:
        """
        Opens a video capture device with the given index and displays its frames in a window.
        The window is closed when the 'q' key is pressed or the given event is set.

        :param index: The index of the video capture device to open.
        :param event: An event object that can be used to signal the method to stop displaying frames. using

        :raises cv2.error: If the video capture device cannot be opened.

        Example:
        Open the video capture device with index 0 and display its frames in a window.
        identify_camera(0, stop_event)
        Wait for the user to press the 'q' key or signal the stop event.
        stop_event.wait()

        """
        cap = (
            cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if self.os_name == "Windows"
            else cv2.VideoCapture(index)
        )
        while not event.is_set():
            ret, frame = cap.read()
            cv2.imshow(f"Camera {index}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_to_json(self, filename: str = "camera_config.json") -> None:
        """
        Saves the camera order with unique identifiers to a JSON file.

        :param filename: The name of the JSON file to save the camera order to.
        """
        # Read existing data
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Update with new data
        data["Camera Order"] = self.camera_mapping

        # Write back to file
        with open(filename, "w") as f:
            json.dump(data, f)

    def get_camera_count(self) -> int:
        """
        Checks to see how many cameras are connected to the computer.

        :return: The number of cameras connected to the computer.
        :rtype: int

        """
        for i in range(self.max_tested):
            cap = (
                cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.os_name == "Windows"
                else cv2.VideoCapture(i)
            )
            if not cap.isOpened():
                cap.release()
                return i
        return self.max_tested

    def identify_all_cameras(self) -> None:
        """
        Allows the user to manually enter a unique identifier for each camera connected to the computer.

        :raises cv2.error: If the video capture device cannot be opened.

        Example:
        Creates a list of unique identifiers for each camera connected to the computer.
        """
        number_cameras = self.get_camera_count()
        for i in range(number_cameras):
            event = Event()
            thread = Thread(target=self.identify_camera, args=(i, event))
            thread.start()
            identifier = input(f"Enter a unique identifier for camera at index {i}: ")
            event.set()
            thread.join()
            if identifier.lower() == "skip":
                if "skip" in self.camera_mapping:
                    self.camera_mapping["skip"].append(i)
                else:
                    self.camera_mapping["skip"] = [i]
            else:
                self.camera_mapping[identifier] = i

    def camera_identifier(self) -> None:
        """
        Checks to see if the camera_config.json file exists and if it contains the 'Camera Order' key.

        If the file does not exist or the 'Camera Order' key is not present, the user is prompted to identify all connected cameras.
        """
        # Check if camera_config.json exists
        if not os.path.exists("camera_config.json"):
            print("camera_config.json not found. Running CameraIdentifier...")
            camera_identifier = CameraIdentifier()
            camera_identifier.identify_all_cameras()
            camera_identifier.save_to_json()
        else:
            # Check if 'Camera Order' exists in the JSON file
            with open("camera_config.json", "r") as f:
                data = json.load(f)
                if "Camera Order" not in data:
                    print(
                        '"Camera Order" not found in camera_config.json. Running CameraIdentifier...'
                    )
                    camera_identifier = CameraIdentifier()
                    camera_identifier.identify_all_cameras()
                    camera_identifier.save_to_json()


# Camera Configurator #


class CameraConfigurator:
    """
    Utility Class for configuring camera exposure and color temperature.
    Sets exposure and color temperature to 0 and 3000 respectively by default.
    If another value is chosen during the setup it will be saved to all connected cameras.
    """

    def __init__(self, device_id: int = 0) -> None:
        self.captureDevice: cv2.VideoCapture = cv2.VideoCapture(
            device_id, cv2.CAP_DSHOW
        )
        self.exposure: int = 0
        self.color_temp: int = 3000
        self.init_camera_settings()

    def init_camera_settings(self) -> None:
        """
        Set camera settings to default values.
        """
        self.captureDevice.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        self.captureDevice.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.color_temp)
        self.captureDevice.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.captureDevice.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    def camera_text_overlay(self, frame: np.ndarray) -> None:
        """
        Adds text overlay to frame with current exposure and color temperature.

        :param frame: The frame to add the text overlay to.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        line_type = 2

        position_exposure = (10, frame.shape[0] - 40)
        cv2.putText(
            frame,
            f"Exposure: {self.exposure}",
            position_exposure,
            font,
            font_scale,
            font_color,
            line_type,
        )

        position_color_temp = (10, frame.shape[0] - 80)
        cv2.putText(
            frame,
            f"Color Temp: {self.color_temp}",
            position_color_temp,
            font,
            font_scale,
            font_color,
            line_type,
        )

    def update_camera_settings(self, key: int) -> None:
        """
        Checks if buttons 1, 2, 4 or 5 are pressed and updates the camera settings accordingly.

        :param key: The key that was pressed.
        """
        if key == ord("2"):
            self.exposure += 1
            self.captureDevice.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        elif key == ord("1"):
            self.exposure -= 1
            self.captureDevice.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        elif key == ord("5"):
            self.color_temp += 50
            self.captureDevice.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.color_temp)
        elif key == ord("4"):
            self.color_temp -= 50
            self.captureDevice.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.color_temp)

    def save_to_json(self, filename: str = "camera_config.json"):
        """
        Saves the camera order with unique identifiers to a JSON file.

        :param filename: The name of the JSON file to save the camera order to.
        """
        # Read existing data
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Update with new data
        data["CameraSettings"] = {
            "Camera Exposure": self.exposure,
            "Camera Color Temperature": self.color_temp,
        }

        # Write back to file
        with open(filename, "w") as f:
            json.dump(data, f)

    def run(self) -> None:
        """
        Runs the camera configurator.
        """
        while self.captureDevice.isOpened():
            ret, frame = self.captureDevice.read()
            key = cv2.waitKey(1)

            self.update_camera_settings(key)
            self.camera_text_overlay(frame)

            cv2.imshow("Webcam", frame)

            if key == ord("q"):
                break

        self.captureDevice.release()
        cv2.destroyAllWindows()

    def camera_configurator(self) -> None:
        """
        Configures the camera settings based on the existing JSON file or creates a new one.
        """
        # Check if camera_settings.json exists
        if not os.path.exists("camera_config.json"):
            print("camera_settings.json not found. Running CameraConfigurator...")
            camera_configurator = CameraConfigurator()
            camera_configurator.run()
            camera_configurator.save_to_json()
        else:
            # Check if 'Camera Exposure' or 'Camera Color Temperature' exists in the JSON file
            with open("camera_config.json", "r") as f:
                data = json.load(f)
                camera_settings = data.get("CameraSettings", {})
                if (
                    "Camera Exposure" not in camera_settings
                    or "Camera Color Temperature" not in camera_settings
                ):
                    print(
                        '"Camera Exposure" or "Camera Color Temperature" not found in camera_config.json. Running CameraConfigurator...'
                    )
                    camera_configurator = CameraConfigurator()
                    camera_configurator.run()
                    camera_configurator.save_to_json()




# File structures #

class Warehouse:
    """
    Class for building a structurized data warehouse for train|test images.
    """

    def __init__(self) -> None:
        self.created_dirs = []
        self.created_sub_dirs = []
        self.created_nested_sub_dirs = []
        self.object_name = ""
        self.anomalies = []  # Used in camera.py

    def clean_folder_name(self, folder_name: str) -> str:
        """
        Replaces spaces in folder name with underscores.

        :param folder_name: The folder name to clean.
        :return: The cleaned folder name.

        Example:
        clean_folder_name("my folder") -> "my_folder"
        """
        cleaned_name = folder_name.replace(" ", "_")
        return cleaned_name

    def create_directory(self, path: str) -> None:
        """
        Create a directory row in the Anvil Data Table and add its name to the created_dirs list.

        Parameters:
            path (str): The path where the directory should be created.
        """
        try:
            # Add a new row to the 'Images' table
            app_tables.images.add_row(directory_name=path)
            self.created_dirs.append(os.path.basename(path))
        except Exception as e:
            print(f"An error occurred while creating {path}: {e}")

    def build(
        self,
        object_name: str,
        anomalies = ["Default1", "Default2", "Default3"],
    ):
        """
        Build the data warehouse directory structure based on the given object name and anomalies.

        Parameters:
            object_name (str): The name of the object directory.
            anomalies (List[str]): A list of anomaly names for nested subdirectories.
        """
        self.object_name = self.clean_folder_name(object_name)
        base_dir_path = "data_warehouse"
        dataset_dir_path = f"{base_dir_path}/dataset"
        object_dir_path = f"{dataset_dir_path}/{self.object_name}"

        for path in [base_dir_path, dataset_dir_path]:
            self.create_directory(path)

        self.create_directory(object_dir_path)

        self.anomalies = anomalies  # populates the anomalies list so that camera.py can use it
        sub_dirs = ["train", "test"]
        nested_sub_dirs = {"train": ["good"], "test": ["good"] + anomalies}

        for sub_dir in sub_dirs:
            sub_dir_path = f"{object_dir_path}/{sub_dir}"
            self.create_directory(sub_dir_path)

            for nested_sub_dir in nested_sub_dirs.get(sub_dir, []):
                nested_sub_dir = self.clean_folder_name(nested_sub_dir)
                nested_sub_dir_path = f"{sub_dir_path}/{nested_sub_dir}"
                self.create_directory(nested_sub_dir_path)

    def __str__(self) -> str:
        ret = ""
        if self.created_dirs:
            ret += f"Created the following directories: {', '.join(self.created_dirs)} in {self.object_name} \n"
        if self.created_sub_dirs:
            ret += f"Created the following subdirectories for object {self.object_name}: {', '.join(self.created_sub_dirs)}"
        if self.created_nested_sub_dirs:
            ret += f"With nested subdirectories: {', '.join(self.created_nested_sub_dirs)}"
        elif not any([self.created_dirs, self.created_sub_dirs, self.created_nested_sub_dirs]):
            ret += f"Directory {self.object_name} already exist! Nothing has been created."
        return ret
