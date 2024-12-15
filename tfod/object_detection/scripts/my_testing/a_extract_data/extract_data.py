from pycocotools.coco import COCO
import cv2
import os
import shutil
from utils.global_vars import paths


def clean_directory(directory):
    """
    Clean a directory by removing all its contents.
    Args:
        directory (str): Path to the directory to clean.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove all contents
    os.makedirs(directory)  # Recreate the directory


def extract_images(max_images_per_category=5000, min_resolution=(128, 128)):
    """
    Extract and crop images from COCO dataset with a minimum resolution filter.

    Args:
        max_images_per_category (int): Maximum number of images to extract per category.
        min_resolution (tuple): Minimum width and height (width, height).
    """
    # Clean the RAW_DIR first
    clean_directory(paths.RAW_DIR)

    # Initialize COCO API
    coco = COCO(paths.ANNOTATIONS)
    categories = ["cat", "dog", "bicycle"]  # Categories to extract
    category_ids = coco.getCatIds(catNms=categories)

    total_images_extracted = 0
    min_width, min_height = min_resolution

    for category_id in category_ids:
        category_name = coco.loadCats(category_id)[0]["name"]
        category_dir = os.path.join(paths.RAW_DIR, category_name)  # Save to RAW_DIR
        os.makedirs(category_dir, exist_ok=True)

        # Get annotations for the category
        ann_ids = coco.getAnnIds(catIds=[category_id])
        annotations = coco.loadAnns(ann_ids)

        print(f"\nExtracting images for category: {category_name}...")
        images_extracted = 0

        for ann in annotations:
            if images_extracted >= max_images_per_category:
                break

            img_info = coco.loadImgs(ann["image_id"])[0]
            img_path = os.path.join(paths.IMAGES_DIR, img_info["file_name"])
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image at {img_path}. Skipping.")
                continue

            # Crop the bounding box
            x, y, w, h = map(int, ann["bbox"])
            if w < min_width or h < min_height:
                print(
                    f"Skipping image {img_info['file_name']} with bbox size ({w}x{h}) - Below minimum resolution."
                )
                continue

            cropped_img = img[y : y + h, x : x + w]

            # Save the cropped image
            output_path = os.path.join(
                category_dir, f"{img_info['id']}_{ann['id']}.jpg"
            )
            cv2.imwrite(output_path, cropped_img)
            images_extracted += 1

        print(f"Extracted {images_extracted} images for category: {category_name}.")
        total_images_extracted += images_extracted

    print(f"\n=== Extraction Complete ===")
    print(f"Total images extracted: {total_images_extracted}")


if __name__ == "__main__":
    extract_images(max_images_per_category=5000, min_resolution=(128, 128))
