import cv2
import torch
from torchvision import transforms, models
import os
from utils.global_vars import paths
import torch.nn as nn


def load_model():
    num_classes = len(os.listdir(paths.TRAIN_DIR))
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )

    try:
        model.load_state_dict(
            torch.load(paths.MODEL_SAVE_PATH, map_location=torch.device("cpu"))
        )
        model.eval()
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
        exit()

    return model


def preprocess_frame(frame):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    try:
        return transform(frame).unsqueeze(0)
    except Exception as e:
        print(f"Error during frame preprocessing: {e}")
        return None


def test_camera():
    cam = cv2.VideoCapture(4)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow("Camera", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
