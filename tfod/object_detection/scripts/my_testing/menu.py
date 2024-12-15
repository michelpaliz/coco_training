import os
import threading
from a_extract_data.extract_data import extract_images
from b_split_data.split_data import split_data
from c_train_model.train_model import train_model
from d_test_model.test_model import test_model
from e_camera.live_camera_test import test_camera
from f_analyses.visualization.confusion_matrix import plot_confusion_matrix
from f_analyses.visualization.metrics import plot_metrics
from f_analyses.visualization.model_arch import visualize_model_architecture


# File to track progress
PROGRESS_FILE = "progress.txt"


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


def handle_train_model():
    try:
        print("\nTraining Model...")
        train_model()
        update_progress("Step 4", "Completed")
    except Exception as e:
        print(f"Error during model training: {e}")
        update_progress("Step 3", "Failed")


def handle_test_model():
    try:
        print("\nTesting Model...")
        test_model()
        update_progress("Step 5", "Completed")
    except Exception as e:
        print(f"Error during model testing: {e}")
        update_progress("Step 4", "Failed")


def handle_plot_metrics():
    try:
        print("\nVisualizing Training Metrics...")
        plot_metrics()
        update_progress("Step 6", "Completed")
    except Exception as e:
        print(f"Error during metrics visualization: {e}")
        update_progress("Step 5", "Failed")


def handle_plot_confusion_matrix():
    try:
        print("\nVisualizing Confusion Matrix...")
        plot_confusion_matrix()
        update_progress("Step 7", "Completed")
    except Exception as e:
        print(f"Error during confusion matrix visualization: {e}")
        update_progress("Step 6", "Failed")


def visualize_model_arch():
    try:
        print("\Testing with your own data.")
        visualize_model_architecture()
        update_progress("Step 8", "Completed")
    except Exception as e:
        print(f"Error during model architecture visualization: {e}")
        update_progress("Step 7", "Failed")


def handle_test_camera():
    try:
        print("\nTesting Model with Camera...")
        test_camera()
        # camera_thread = threading.Thread(target=test_camera)
        # camera_thread.start()
        # camera_thread.join()
        update_progress("Step 8", "Completed")
    except Exception as e:
        print(f"Error during camera test: {e}")
        update_progress("Step 8", "Failed")


# Main Menu
def main_menu():
    menu_actions = {
        "1": handle_extract_data,
        "2": handle_split_data,
        "3": handle_train_model,
        "4": handle_test_model,
        "5": handle_plot_metrics,
        "6": handle_plot_confusion_matrix,
        "7": visualize_model_arch,
        "8": handle_test_camera,
        "9": exit,
    }

    while True:
        print("\n=== Main Menu ===")
        print("1. Extract Data (Crop images from COCO)")
        print("2. Split Data (Train/Validation split)")
        print("3. Train Model")
        print("4. Test Model")
        print("5. Visualize Training Metrics (Accuracy/Loss Curve)")
        print("6. Visualize Confusion Matrix")
        print("7. Visualize Model Architecture")
        print("8.Test Model with Camera")
        print("9. Exit")

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
            file.write("Step 9 - Not applicable\n")

    main_menu()
