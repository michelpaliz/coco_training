import cv2
import os
from utils.global_vars import paths


def resize_images(clean_directories):
    """
    Resize images to a standard size for training and validation.

    Args:
        clean_directories (bool): If True, clean directories before resizing.
    """
    size = (128, 128)  # Target size for resizing
    total_resized_images = 0

    if clean_directories:
        print("\nCleaning directories...")
        for directory in [paths.TRAIN_DIR, paths.VAL_DIR]:
            for category in os.listdir(directory):
                category_path = os.path.join(directory, category)
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    os.remove(file_path)
                print(f"  - Cleaned directory: {category_path}")

    print("\n=== Resizing Images ===")
    for directory in [paths.TRAIN_DIR, paths.VAL_DIR]:
        directory_name = (
            "Training Set" if directory == paths.TRAIN_DIR else "Validation Set"
        )
        print(f"\nProcessing {directory_name}...")

        for category in os.listdir(directory):
            category_path = os.path.join(directory, category)
            resized_count = 0

            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Could not read image at {img_path}. Skipping.")
                    continue

                # Check if the image is already resized
                if img.shape[:2] == size:
                    continue

                # Resize the image
                resized_img = cv2.resize(img, size)

                # Save the resized image
                cv2.imwrite(img_path, resized_img)
                resized_count += 1

            # Progress feedback for current category
            print(f"  - Resized {resized_count} images in category '{category}'.")
            total_resized_images += resized_count

    # Final summary
    print("\n=== Resizing Complete ===")
    print(f"Total resized images: {total_resized_images}")
