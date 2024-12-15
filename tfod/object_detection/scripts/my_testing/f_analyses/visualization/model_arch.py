import torch
from torchvision import models
import torch.nn as nn
import os
from utils.global_vars import paths


def visualize_model_architecture():
    try:
        # Load the saved model
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),  # Example architecture
            nn.ReLU(),
            nn.Linear(
                512, len(os.listdir(paths.TRAIN_DIR))
            ),  # Adjust number of classes
        )

        # Load state dict with strict=False
        model.load_state_dict(
            torch.load(paths.MODEL_SAVE_PATH, map_location=torch.device("cpu")),
            strict=False,
        )

        print("Model Architecture:")
        print(model)
    except Exception as e:
        print(f"Error during model architecture visualization: {e}")


if __name__ == "__main__":
    visualize_model_architecture()
