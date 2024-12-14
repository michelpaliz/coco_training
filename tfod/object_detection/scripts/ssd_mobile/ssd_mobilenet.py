import cv2
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

# Load the pre-trained SSD MobileNet model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Define COCO category labels for animals
category_labels = {
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
}


def detect_objects(frame, model):
    # Ensure the input tensor is of dtype uint8 (as expected by the model)
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)[tf.newaxis, ...]
    # Run the model on the input tensor
    outputs = model(input_tensor)
    detections = {key: value.numpy() for key, value in outputs.items()}
    return detections


def draw_boxes(frame, detections, threshold=0.5):
    height, width, _ = frame.shape

    # Loop through each detection
    for i in range(int(detections["num_detections"])):
        score = detections["detection_scores"][0][i]  # Get the i-th score
        if score < threshold:
            continue

        # Get bounding box coordinates
        bbox = detections["detection_boxes"][0][i]  # Get the i-th bounding box
        ymin, xmin, ymax, xmax = bbox
        left, top, right, bottom = (
            int(xmin * width),
            int(ymin * height),
            int(xmax * width),
            int(ymax * height),
        )

        # Get the class label
        class_id = int(detections["detection_classes"][0][i])  # Get the i-th class
        label = category_labels.get(class_id, "Unknown")

        # Draw the bounding box and label
        if label != "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return frame


# Open the webcam
cap = cv2.VideoCapture(11)

print("Starting live camera... Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame for a mirror effect (optional)
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB (TensorFlow model expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect objects in the frame
    detections = detect_objects(rgb_frame, model)
    # Draw bounding boxes for detected animals
    frame = draw_boxes(frame, detections, threshold=0.5)
    # Display the frame
    cv2.imshow("Live Animal Detection", frame)
    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

