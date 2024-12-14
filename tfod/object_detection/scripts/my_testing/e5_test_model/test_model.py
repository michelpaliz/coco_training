import os
import torch
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from utils.global_vars import paths


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing for test images
    transform = transforms.Compose(
        [
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(model.fc.in_features, len(os.listdir(paths.TRAIN_DIR))),
    )
    model.load_state_dict(torch.load(paths.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Testing dataset
    y_true, y_pred = [], []
    class_names = os.listdir(paths.TRAIN_DIR)

    correct_predictions = 0
    total_predictions = 0

    print("\n=== Per Image Analysis ===")
    for category in os.listdir(paths.TEST_DIR):
        category_path = os.path.join(paths.TEST_DIR, category)
        if not os.path.isdir(category_path):
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = class_names[predicted_idx.item()]

            # Append results for metrics
            y_true.append(category)
            y_pred.append(predicted_class)

            # Check correctness and display message
            is_correct = category == predicted_class
            correct_predictions += int(is_correct)
            total_predictions += 1
            correctness = "Correct" if is_correct else "Incorrect"
            print(
                f"Image: {img_name}, True Label: {category}, Predicted Label: {predicted_class}, "
                f"Prediction: {correctness}"
            )

    # Summary of correct and incorrect predictions
    incorrect_predictions = total_predictions - correct_predictions
    print("\n=== Prediction Summary ===")
    print(f"Total Images: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {incorrect_predictions}")

    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.show()


if __name__ == "__main__":
    test_model()
