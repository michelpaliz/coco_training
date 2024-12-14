from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


def plot_confusion_matrix():
    # Load saved confusion matrix data
    with open("confusion_matrix_data.pkl", "rb") as f:
        y_true, y_pred, class_names = pickle.load(f)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)

    # Add titles and labels
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix()
