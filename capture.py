from multicamcomposepro.augment import DataAugmenter
from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import CameraConfigurator, Warehouse


def main():
    # Create a structured data warehouse
    warehouse = Warehouse()
    warehouse.build()
    print(warehouse)

    camera_manager = CameraManager(warehouse)
    camera_manager.run()
    print(warehouse)


if __name__ == "__main__":
    main()
