import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils.global_vars import paths


def train_model(num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train a model using the preprocessed and filtered dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation and Preprocessing
    train_transform = transforms.Compose([
        transforms.Resize(144),  # Resize shorter side to 144
        transforms.RandomCrop(128),  # Random crop to 128x128
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(paths.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(paths.VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust output size
    )
    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training Phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_dir = os.path.dirname(paths.MODEL_SAVE_PATH)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.state_dict(), paths.MODEL_SAVE_PATH)
            print(f"Best model saved at {paths.MODEL_SAVE_PATH}")

        # Step the scheduler
        scheduler.step()

    print("\nTraining complete.")


if __name__ == "__main__":
    train_model(num_epochs=20, batch_size=32, learning_rate=0.001)
