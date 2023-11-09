from multicamcomposepro.augment import DataAugmenter
from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import CameraConfigurator, Warehouse


def main():
    # Create a structured data warehouse
    warehouse = Warehouse()
    warehouse.build()
    print(warehouse)
    # cc = CameraConfigurator()

    camera_manager = CameraManager(
        warehouse, test_anomaly_images=0, train_images=1, allow_user_input=False
    )
    camera_manager.run()
    print(warehouse)


if __name__ == "__main__":
    main()
