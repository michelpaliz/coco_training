class Paths:
    # Define the absolute base path to your USB
    BASE_DIR = "/media/michael/C60D-9867/COCO"

    # Raw COCO dataset
    ANNOTATIONS = f"{BASE_DIR}/annotations/instances_train2017.json"
    IMAGES_DIR = f"{BASE_DIR}/train2017/train2017"

    # Processed dataset
    RAW_DIR = f"{BASE_DIR}/raw"  # Add RAW_DIR for initial extracted images
    PROCESSED_DIR = f"{BASE_DIR}/processed"
    TRAIN_DIR = f"{PROCESSED_DIR}/train"
    VAL_DIR = f"{PROCESSED_DIR}/val"
    # Test dataset
    TEST_DIR = f"{BASE_DIR}/test"

    # Model directory
    MODEL_SAVE_PATH = f"{BASE_DIR}/models/model.pth"


# Instantiate the global paths object
paths = Paths()
