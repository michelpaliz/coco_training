import cv2
import torch
from torchvision import transforms, models
import os
from utils.global_vars import paths
import torch.nn as nn


def load_model():
    """
    Load the trained model with the same architecture used during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the architecture
    num_classes = len(os.listdir(paths.TRAIN_DIR))
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )

    # Load the trained model
    try:
        model.load_state_dict(torch.load(paths.MODEL_SAVE_PATH, map_location=device))
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    return model, device


def preprocess_frame(frame):
    """
    Preprocess a camera frame to match the input requirements of the model.
    """
    try:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(144),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(frame).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error during frame preprocessing: {e}")
        return None


def test_camera():
    """
    Use the camera feed to capture frames and classify them using the trained model.
    """
    cap = cv2.VideoCapture(0)  # Adjust camera index if needed

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    model, device = load_model()
    class_names = os.listdir(paths.TRAIN_DIR)

    print("Starting camera. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read a frame from the camera.")
            break

        # Preprocess frame
        input_tensor = preprocess_frame(frame)
        if input_tensor is None:
            print("Skipping frame due to preprocessing error.")
            continue

        # Make prediction
        try:
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = class_names[predicted_idx.item()]

            # Overlay prediction on the frame
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Display the frame
            cv2.imshow("Live Prediction", frame)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            continue

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Camera test terminated by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
