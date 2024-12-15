import pickle
import matplotlib.pyplot as plt

def plot_metrics():
    # Load metrics from the training process
    with open("training_metrics.pkl", "rb") as f:
        train_losses, val_losses, train_accuracies, val_accuracies = pickle.load(f)

    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics()
