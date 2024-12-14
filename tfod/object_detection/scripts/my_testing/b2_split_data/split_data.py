import os
import random
import shutil
from utils.global_vars import paths


def clean_directory(directory):
    """
    Removes all contents of a directory to ensure a clean start.
    """
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))


def split_data():
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    total_train_images = 0
    total_val_images = 0
    total_test_images = 0

    print("\n=== Cleaning Directories ===")
    clean_directory(paths.TRAIN_DIR)
    print(f"Cleaned directory: {paths.TRAIN_DIR}")
    clean_directory(paths.VAL_DIR)
    print(f"Cleaned directory: {paths.VAL_DIR}")
    clean_directory(paths.TEST_DIR)
    print(f"Cleaned directory: {paths.TEST_DIR}")

    print("\n=== Splitting Data into Train, Validation, and Test Sets ===")
    for category in os.listdir(paths.RAW_DIR):  # Use RAW_DIR as source
        category_path = os.path.join(paths.RAW_DIR, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Ensure output directories exist
        os.makedirs(os.path.join(paths.TRAIN_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(paths.VAL_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(paths.TEST_DIR, category), exist_ok=True)

        # Move images
        for img in train_images:
            shutil.copy(
                os.path.join(category_path, img),
                os.path.join(paths.TRAIN_DIR, category, img),
            )
        for img in val_images:
            shutil.copy(
                os.path.join(category_path, img),
                os.path.join(paths.VAL_DIR, category, img),
            )
        for img in test_images:
            shutil.copy(
                os.path.join(category_path, img),
                os.path.join(paths.TEST_DIR, category, img),
            )

        print(f"Category '{category}':")
        print(f"  - {len(train_images)} images moved to training set.")
        print(f"  - {len(val_images)} images moved to validation set.")
        print(f"  - {len(test_images)} images moved to testing set.")

        total_train_images += len(train_images)
        total_val_images += len(val_images)
        total_test_images += len(test_images)

    print("\n=== Data Splitting Complete ===")
    print(f"Total images in training set: {total_train_images}")
    print(f"Total images in validation set: {total_val_images}")
    print(f"Total images in testing set: {total_test_images}")


if __name__ == "__main__":
    split_data()
