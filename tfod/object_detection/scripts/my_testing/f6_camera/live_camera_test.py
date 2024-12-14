import cv2
import torch
from torchvision import transforms
import numpy as np
import os
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from utils.global_vars import paths


# Load the trained PyTorch model
def load_model():
    print("Loading custom PyTorch model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Update the last layer to match the number of classes in your training data
    model.fc = torch.nn.Linear(model.fc.in_features, len(os.listdir(paths.TRAIN_DIR)))

    try:
        model.load_state_dict(
            torch.load(paths.MODEL_SAVE_PATH, map_location=torch.device("cpu"))
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    model.eval()
    return model


# Function to preprocess frames
def preprocess_frame(frame):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(frame).unsqueeze(0)  # Add batch dimension


# Function to perform object detection
def detect_objects(frame, model, class_names):
    # Preprocess the frame
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = class_names[predicted_idx.item()]
    return predicted_class, confidence.item()


# Draw prediction on the frame
def draw_prediction(frame, predicted_class, confidence):
    label = f"{predicted_class} ({confidence:.2f})"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


# Main camera testing function
def test_camera():
    print("Initializing camera test...")

    # Load the trained model
    model = load_model()

    # Load class names from training directory
    class_names = os.listdir(paths.TRAIN_DIR)

    # Open the webcam
    cap = cv2.VideoCapture(9)  # Replace with the correct camera index
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Starting live camera testing. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Convert BGR frame to RGB for PyTorch
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects in the frame
        predicted_class, confidence = detect_objects(rgb_frame, model, class_names)

        # Draw the prediction on the frame
        frame = draw_prediction(frame, predicted_class, confidence)

        # Display the frame
        cv2.imshow("Live Object Detection", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test ended.")


