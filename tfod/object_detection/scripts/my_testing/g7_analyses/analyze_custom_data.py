import torch
from torchvision import transforms, models
from PIL import Image
import os
import pickle
from utils.global_vars import paths


def test_costume_data():
    # Directory containing test images
    test_dir = paths.OWN_DATA_DIR

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist.")
        return
    img_size = (128, 128)
    # Image transformation
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # Resize shorter side to the final input size
            transforms.CenterCrop(128),  # Center crop to match training input size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard normalization
        ]
    )

    # Define the model architecture
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(os.listdir(paths.TRAIN_DIR))),
    )

    # Load the trained model state_dict
    try:
        model.load_state_dict(
            torch.load(paths.MODEL_SAVE_PATH, map_location=torch.device("cpu"))
        )
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # Class names
    class_names = os.listdir(paths.TRAIN_DIR)

    print("\n=== Testing Model on Unseen Data ===")
    prediction_counts = {class_name: 0 for class_name in class_names}
    correct_predictions = 0
    total_images = 0

    # Data for confusion matrix
    y_true = []
    y_pred = []

    # Traverse subdirectories
    for category in os.listdir(test_dir):
        category_path = os.path.join(test_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        print(f"\nProcessing category: {category}")
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            try:
                # Open and preprocess the image
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

                # Predict the class
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    predicted_class = class_names[predicted_idx.item()]

                    # Append true and predicted labels for confusion matrix
                    y_true.append(class_names.index(category))
                    y_pred.append(predicted_idx.item())

                    # Track correct predictions
                    if predicted_class == category:
                        correct_predictions += 1

                    prediction_counts[predicted_class] += 1

                # Print detailed prediction
                print(
                    f"Image: {img_name}, True Class: {category}, Predicted Class: {predicted_class} ({confidence:.2f})"
                )

                total_images += 1

            except Exception as e:
                print(f"Warning: Could not process image '{img_name}'. Error: {e}")

    # Testing summary
    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    print("\n=== Testing Summary ===")
    print(f"Total images processed: {total_images}")
    print(f"Correctly classified: {correct_predictions} ({accuracy:.2f}%)")
    print("Predictions breakdown:")
    for class_name, count in prediction_counts.items():
        print(f"  {class_name}: {count} images")

    # Save confusion matrix data
    with open("confusion_matrix_data.pkl", "wb") as f:
        pickle.dump((y_true, y_pred, class_names), f)
    print("\nConfusion matrix data saved to 'confusion_matrix_data.pkl'")


if __name__ == "__main__":
    test_costume_data()
