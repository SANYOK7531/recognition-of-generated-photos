# README: Image Recognition Model

## 📌 Project Overview
This project implements a deep learning model for recognition real and generated images using TensorFlow/Keras, structured with modular components and configurable paths.

## 🚀 Dependencies
- **numpy (np):** Numerical operations with arrays.
- **cv2 (OpenCV):** Image loading and preprocessing.
- **sklearn.model_selection:** Data splitting.
- **tensorflow.keras:** Model building and training.
- **matplotlib:** Training performance visualization.
- **seaborn:** Confusion matrix plotting.
- **config.py:** Dataset and model path configurations.
- **images_loader.py:** Custom image loading module.

## 📂 Data Preprocessing
- `ImagesLoader` loads images from configured paths.
- Images are resized to **100x100 px** and normalized to **[0, 1]**.

## 🧩 Dataset Creation
- Labels: `0` (real), `1` (generated)
- **Data Split:**
  - Training: **70%**
  - Validation: **12%**
  - Test: **18%**

## 🏗️ Model Architecture
- **Preprocessing:** Rescaling, random flipping, zooming, and contrast adjustments.
- **Convolutional Layers:**
  - 3 `Conv2D` layers with ReLU activation (3x3 kernels)
  - 3 `MaxPooling2D` layers
- **Fully Connected Layers:**
  - `Flatten()`
  - `Dense(128)` with ReLU
  - `Dropout(0.5)`
  - `Dense(1)` with sigmoid activation

## 🛠️ Training
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Epochs:** 60
- **Batch Size:** 32

## 📊 Evaluation
- **Metrics:** Test loss and accuracy
- **Plots:**
  - Accuracy trends (using `matplotlib`)
  - Confusion matrix (using `seaborn`)

## 💾 Model Saving
The trained model is saved to the path set in `config.py`.
