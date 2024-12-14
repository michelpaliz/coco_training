import os
import threading
from a1_extract_data.extract_data import extract_images
from b2_split_data.split_data import split_data
from c3_resize_data.resize_data import resize_images
from d4_train_model.train_model import train_model
from e5_test_model.test_model import test_model
from f6_camera.live_camera_test import test_camera
from g7_analyses.visualization.confusion_matrix import plot_confusion_matrix
from g7_analyses.visualization.metrics import plot_metrics
from g7_analyses import analyze_custom_data

# File to track progress
PROGRESS_FILE = "progress.txt"


# Use Cases:

#     Fresh Pipeline Run:
#         If you’re starting over, set clean_directories=True to ensure directories are empty before populating and resizing.
#     Incremental Runs:
#         For incremental updates or debugging, set clean_directories=False to avoid clearing existing processed files.


# Function to update progress
def update_progress(step, status):
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as file:
            for line in file:
                step_name, step_status = line.strip().split(" - ")
                progress[step_name] = step_status

    progress[step] = status

    with open(PROGRESS_FILE, "w") as file:
        for step_name, step_status in progress.items():
            file.write(f"{step_name} - {step_status}\n")


# Functions for menu actions
def handle_extract_data():
    try:
        print("\nExtracting Data...")
        extract_images()
        update_progress("Step 1", "Completed")
    except Exception as e:
        print(f"Error during data extraction: {e}")
        update_progress("Step 1", "Failed")


def handle_split_data():
    try:
        print("\nSplitting Data...")
        split_data()
        update_progress("Step 2", "Completed")
    except Exception as e:
        print(f"Error during data splitting: {e}")
        update_progress("Step 2", "Failed")


def handle_resize_data():
    try:
        print("\nResizing Data...")
        # Ask user if they want to clean directories
        clean_choice = (
            input("Do you want to clean the directories before resizing? (y/n): ")
            .strip()
            .lower()
        )
        clean_flag = (
            clean_choice == "y"
        )  # True if the user chooses 'y', otherwise False

        resize_images(clean_directories=clean_flag)
        update_progress("Step 3", "Completed")
    except Exception as e:
        print(f"Error during data resizing: {e}")
        update_progress("Step 3", "Failed")


def handle_train_model():
    try:
        print("\nTraining Model...")
        train_model()
        update_progress("Step 4", "Completed")
    except Exception as e:
        print(f"Error during model training: {e}")
        update_progress("Step 4", "Failed")


def handle_test_model():
    try:
        print("\nTesting Model...")
        test_model()
        update_progress("Step 5", "Completed")
    except Exception as e:
        print(f"Error during model testing: {e}")
        update_progress("Step 5", "Failed")


def handle_plot_metrics():
    try:
        print("\nVisualizing Training Metrics...")
        plot_metrics()
        update_progress("Step 6", "Completed")
    except Exception as e:
        print(f"Error during metrics visualization: {e}")
        update_progress("Step 6", "Failed")


def handle_plot_confusion_matrix():
    try:
        print("\nVisualizing Confusion Matrix...")
        plot_confusion_matrix()
        update_progress("Step 7", "Completed")
    except Exception as e:
        print(f"Error during confusion matrix visualization: {e}")
        update_progress("Step 7", "Failed")


def test_costume_data():
    try:
        print("\Testing with your own data.")
        analyze_custom_data()
        update_progress("Step 8", "Completed")
    except Exception as e:
        print(f"Error during costume data test: {e}")
        update_progress("Step 8", "Failed")


def handle_test_camera():
    try:
        print("\nTesting Model with Camera...")
        camera_thread = threading.Thread(target=test_camera)
        camera_thread.start()
        camera_thread.join()
        update_progress("Step 9", "Completed")
    except Exception as e:
        print(f"Error during camera test: {e}")
        update_progress("Step 9", "Failed")


# Main Menu
def main_menu():
    menu_actions = {
        "1": handle_extract_data,
        "2": handle_split_data,
        "3": handle_resize_data,
        "4": handle_train_model,
        "5": handle_test_model,
        "6": handle_plot_metrics,
        "7": handle_plot_confusion_matrix,
        "8": test_costume_data,
        "9": handle_test_camera,
        "10": exit,
    }

    while True:
        print("\n=== Main Menu ===")
        print("1. Extract Data (Crop images from COCO)")
        print("2. Split Data (Train/Validation split)")
        print("3. Resize Data (Standardize image size)")
        print("4. Train Model")
        print("5. Test Model")
        print("6. Visualize Training Metrics (Accuracy/Loss Curve)")
        print("7. Visualize Confusion Matrix")
        print("8. Test Model with costume data")
        print("9.Test Model with Camera")
        print("10. Exit")

        choice = input("Enter your choice: ")

        action = menu_actions.get(choice)
        if action:
            action()
        else:
            print("\nInvalid choice. Please try again.")


# Initialize progress file and start menu
if __name__ == "__main__":
    if not os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "w") as file:
            file.write("Step 1 - Not started\n")
            file.write("Step 2 - Not started\n")
            file.write("Step 3 - Not started\n")
            file.write("Step 4 - Not started\n")
            file.write("Step 5 - Not started\n")
            file.write("Step 6 - Not started\n")
            file.write("Step 7 - Not started\n")
            file.write("Step 8 - Not started\n")
            file.write("Step 9 - Not started\n")
            file.write("Step 10 - Not applicable\n")

    main_menu()
