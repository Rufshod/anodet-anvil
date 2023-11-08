from multicamcomposepro.augment import DataAugmenter
from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import CameraConfigurator, Warehouse


def main():
    # Create a structured data warehouse
    warehouse = Warehouse()
    warehouse.build()
    print(warehouse)
    #cc = CameraConfigurator()

    camera_manager = CameraManager(warehouse, test_anomaly_images=0, train_images=1, allow_user_input=False)
    camera_manager.run()
    print(warehouse)


if __name__ == "__main__":
    main()

#######################################################
from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):
    def __init__(self, **properties):
        # Set Form properties and Data Bindings.
        self.init_components(**properties)
        self.image_name = "000.png"  # Assuming this is a default or placeholder image
        # Any code you write here will run before the form opens.

    def button_1_click(self, **event_args):
        """This method is called when the button is clicked"""
        # Update the angle_name from the text box each time the button is clicked
        angle_name = self.text_box_1.text
        if angle_name:  # Check if the text box is not empty
            image_url = anvil.server.call('get_image_url', angle_name, self.image_name)
            self.image_1.source = image_url
        else:
            # Optionally, handle the case where the angle name is empty
            anvil.alert("Please enter an angle name.")

@anvil.server.callable
def get_image_url(angle, image_name):
    image_path = f'http://127.0.0.1:5000/{angle}/{image_name}'
    return image_path