
### **Workflow**

#### **Step 1: Extract Data**
1. **Purpose**: Extract specific categories (e.g., `mouse`, `book`, `keyboard`) from the COCO dataset.
2. **Paths Involved**: 
   - Input: `ANNOTATIONS` and `IMAGES_DIR`.
   - Output: `RAW_DIR` (organized by category).
3. **Function**: `extract_images()`.

```python
# Example Execution
extract_images(max_images_per_category=200)
```
- **Output**: Cropped and categorized images saved in `RAW_DIR`:
  ```
  RAW_DIR/
      mouse/
          img1.jpg
          img2.jpg
      book/
          img1.jpg
          img2.jpg
  ```

---

#### **Step 2: Split Data**
1. **Purpose**: Split the data into `train`, `val`, and `test` sets.
2. **Paths Involved**:
   - Input: `RAW_DIR`.
   - Output: `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR`.
3. **Function**: `split_data()`.

```python
# Example Execution
split_data()
```
- **Output**: The data is moved to respective folders:
  ```
  TRAIN_DIR/
      mouse/
          img1.jpg
          img2.jpg
      book/
  VAL_DIR/
  TEST_DIR/
  ```

---

#### **Step 3: Resize Images**
1. **Purpose**: Ensure all images are resized to a consistent size (e.g., 128x128).
2. **Paths Involved**:
   - Input: `TRAIN_DIR`, `VAL_DIR`.
   - Output: Resized images saved back into `TRAIN_DIR`, `VAL_DIR`.
3. **Function**: `resize_images()`.

```python
# Example Execution
resize_images()
```
- **Output**: Resized images overwrite existing ones in the same directories.

---

#### **Step 4: Train Model**
1. **Purpose**: Train the model using the processed `train` and `val` datasets.
2. **Paths Involved**:
   - Input: `TRAIN_DIR`, `VAL_DIR`.
   - Output: `MODEL_SAVE_PATH` (saved trained model weights).
3. **Function**: `train_model()`.

```python
# Example Execution
train_model(num_epochs=10, batch_size=32, learning_rate=0.001)
```
- **Output**: 
  - Model is saved at `MODEL_SAVE_PATH`.
  - Metrics saved to `training_metrics.pkl`.

---

#### **Step 5: Evaluate Model**
1. **Purpose**: Visualize training/validation performance.
2. **Paths Involved**:
   - Input: `training_metrics.pkl`.
   - Output: Loss/accuracy curves (displayed as graphs).
3. **Function**: `plot_metrics()`.

```python
# Example Execution
plot_metrics()
```
- **Output**: Graphs showing loss and accuracy over epochs.

---

#### **Step 6: Test Model**
1. **Purpose**: Evaluate the model on unseen test data.
2. **Paths Involved**:
   - Input: `TEST_DIR`, `MODEL_SAVE_PATH`.
   - Output: Performance metrics and confusion matrix data.
3. **Function**: `test_model()`.

```python
# Example Execution
test_model()
```
- **Output**:
  - Classification performance on test images.
  - Breakdown of predictions and accuracy.

---

#### **Step 7: Visualize Confusion Matrix**
1. **Purpose**: Visualize the model's predictions vs true labels.
2. **Paths Involved**:
   - Input: True/predicted labels from `test_model()`.
3. **Function**: `plot_confusion_matrix()`.

```python
# Example Execution
plot_confusion_matrix(y_true, y_pred, class_names=["mouse", "keyboard", "book"])
```
- **Output**: Confusion matrix displayed as a heatmap.

---

#### **Step 8: Live Camera Test**
1. **Purpose**: Test the model in real-time using a camera feed.
2. **Paths Involved**:
   - Input: `MODEL_SAVE_PATH` (to load the trained model).
3. **Function**: `test_camera()`.

```python
# Example Execution
test_camera()
```
- **Output**:
  - Real-time classification of objects in the camera feed.
  - Predictions displayed on the video stream.

---

### **Complete Workflow**
Here’s how the workflow looks end-to-end:
1. Extract data: `extract_images()`.
2. Split data: `split_data()`.
3. Resize images: `resize_images()`.
4. Train model: `train_model()`.
5. Evaluate training performance: `plot_metrics()`.
6. Test model on unseen data: `test_model()`.
7. Visualize confusion matrix: `plot_confusion_matrix()`.
8. Test model in real-time: `test_camera()`.

This workflow ensures:
- A systematic approach to building, training, and testing your model.
- Intermediate steps like visualization and analysis for debugging and evaluation.