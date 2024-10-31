
# Hand Gesture-Based Math Operation Detection

This project enables users to detect hand gestures representing numbers and basic mathematical operators (+, -, *, /) 
in real-time using a webcam. The system uses these gestures to perform arithmetic operations, displaying the result.
![Hand Gestures](/image.jpg)

---

## Project Overview

The project is built using Python and relies on computer vision and machine learning techniques to recognize hand gestures. 
It consists of the following main steps:

1. **Data Collection** - Capturing images of hand gestures representing numbers 0-9.
2. **Dataset Preparation** - Preprocessing the collected images for training.
3. **Model Training** - Training a classifier to recognize hand gestures.
4. **Real-Time Detection** - Detecting gestures in real-time and performing arithmetic operations.

---

## Prerequisites

1. **Python Version:** Ensure you have Python 3.7 or higher installed.
2. **Required Libraries:** The following libraries are necessary to run this project:
   - OpenCV
   - MediaPipe
   - NumPy
   - Scikit-Learn

   To install all required libraries, use the following command:

   ```bash
   pip install -r requirements.txt
   ```

---

## Step-by-Step Execution Guide

### 1. Data Collection

Before training, create a dataset of hand gestures representing numbers 0 through 9. This is achieved using the `collectimages.py` script.

**File:** `collectimages.py`

- **Functionality:** This script captures images from your webcam for each number (0-9), saving each image in a specific directory.
- **Usage:** Run the script, and it will guide you through capturing 100 images for each hand gesture.
- **Steps to Run:**
  - Ensure your webcam is active and that you can position your hand gestures clearly within the camera frame.
  - Run the following command:

  ```bash
  python collectimages.py
  ```

- **Process:** For each number:
  - You’ll be prompted with the message "Are you ready?" in the camera window.
  - Press 'q' to start capturing images for the displayed number.
  - Hold each gesture steadily while images are captured.

- **Directory Structure:** The captured images should be organized in the following structure:

```
./dataset/
    ├── 0/      # Folder for gesture representing "0"
    ├── 1/      # Folder for gesture representing "1"
    ├── ...
    └── 9/      # Folder for gesture representing "9"
```

Ensure that each folder contains 100 images of the respective hand gesture.

---

### 2. Dataset Preparation

After capturing images, you need to preprocess the data to make it suitable for model training. Use the `createdataset.py` script.

**File:** `createdataset.py`

- **Functionality:** This script loads and processes images from each folder in the dataset directory. It normalizes the data and prepares it in a structured format, then saves it as a serialized file for model training.
- **Steps to Run:** Execute the following command:

  ```bash
  python createdataset.py
  ```

- **Output:** This script outputs a processed dataset file, which is used for training the model.

---

### 3. Model Training

Once the dataset is prepared, train the classifier model using `trainclassifier.py`. This script uses a Random Forest classifier to recognize hand gestures.

**File:** `trainclassifier.py`

- **Functionality:** This script loads the preprocessed dataset, trains a Random Forest model, and saves the trained model as `rf_model.p`.
- **Random Forest Model:** This model is well-suited for multi-class classification problems like gesture recognition.
- **Steps to Run:** Execute the following command:

  ```bash
  python trainclassifier.py
  ```

- **Output:** A file named `rf_model.p`, containing the trained Random Forest model.

---

### 4. Real-Time Detection and Math Operation

After training the model, use `main.py` for real-time gesture detection and performing mathematical operations based on detected gestures.

**File:** `main.py`

- **Functionality:** This script uses the trained model to recognize hand gestures in real-time through a webcam. It displays the recognized gestures and performs arithmetic operations based on the detected numbers and operator.
- **Process Overview:**
  - **Hand Detection:** Uses MediaPipe to detect hand landmarks.
  - **Gesture Recognition:** Classifies detected hand gestures as numbers or operators using the trained model.
  - **Operation Flow:**
      - Detects the first number.
      - Allows selection of an operator by hovering the index finger over one of the displayed operator symbols (+, -, *, /).
      - Detects the second number.
      - Calculates and displays the result for 5 seconds before resetting for a new operation.
- **Steps to Run:** Execute the following command:

  ```bash
  python main.py
  ```

---

## Model Information

This project uses a **Random Forest Classifier** trained on images of hand gestures representing numbers 0-9.

### Key Components:

1. **Data Preprocessing:** Images are resized and normalized to improve model accuracy.
2. **Model Training:** The Random Forest classifier uses multiple decision trees to classify gestures based on landmark features.
3. **Hand Tracking:** MediaPipe tracks hand landmarks in real-time, which are then fed into the model for classification.
4. **Real-Time Detection:** The script processes video frames, applies the trained model, and performs arithmetic operations based on recognized gestures.

---

## Project Workflow Summary

1. **Capture Images:** `collectimages.py` - Collect 100 images for each gesture.
2. **Prepare Dataset:** `createdataset.py` - Normalize and structure images.
3. **Train Model:** `trainclassifier.py` - Train a Random Forest model on the prepared dataset.
4. **Real-Time Detection:** `main.py` - Detect gestures in real-time and perform arithmetic operations.

---

## Licence

This project is licensed under the terms of the [Licencia Apache 2.0](LICENSE).

---

## Troubleshooting

- Ensure that each script is run in sequence as described.
- If encountering issues with MediaPipe, verify the camera permission settings.
- For problems with OpenCV display, check if you have installed all required dependencies.

For further questions, refer to official documentation for each library.
