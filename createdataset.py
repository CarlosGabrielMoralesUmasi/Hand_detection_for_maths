import os 
import pickle 

import cv2
import mediapipe as mp

# Import MediaPipe's hands module for hand detection and landmark estimation
mp_hands = mp.solutions.hands

# Initialize the Hands object from MediaPipe
# - static_image_mode=True: Indicates that the input will be static images (not video stream)
# - min_detection_confidence=0.3: Minimum confidence level required to detect a hand
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './dataset'
dataset = []
labels = []

# Loop through each directory (representing each class) inside the dataset folder
for directory in os.listdir(data_dir):
     path = os.path.join(data_dir, directory)  # Construct the full path for the current class directory

    # Loop through each image file in the current class directory
     for img_path in os.listdir(path):
        normalized_landmarks = []  # List to store normalized x, y coordinates
        x_coordinates, y_coordinates = [], []  # Temporary lists for x and y coordinates

        # Read the image and convert it from BGR to RGB format (required by MediaPipe)
        image = cv2.imread(os.path.join(path, img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands using MediaPipe's hand processing method
        processed_image = hands.process(image_rgb)

        # Get the hand landmarks (if any hand is detected in the image)
        hand_landmarks = processed_image.multi_hand_landmarks

        if hand_landmarks:  # If hand landmarks are found
            for hand_landmark in hand_landmarks:
                landmark_coordinates = hand_landmark.landmark  # Get individual landmark coordinates

                # Extract the x and y coordinates of all landmarks
                for coordinates in landmark_coordinates:
                    x_coordinates.append(coordinates.x)
                    y_coordinates.append(coordinates.y)

                # Find the minimum x and y values to normalize the coordinates
                min_x, min_y = min(x_coordinates), min(y_coordinates)
                 # Normalize the landmarks by subtracting the minimum x and y values
                for coordinates in landmark_coordinates:
                    normalized_x = coordinates.x - min_x
                    normalized_y = coordinates.y - min_y
                    normalized_landmarks.extend((normalized_x, normalized_y))  # Add normalized values to the list

            # Append the normalized landmarks to the dataset
            dataset.append(normalized_landmarks)

            # Append the label (class name) for the current directory
            labels.append(directory)
# Open (or create) a file called 'extracted_landmarks.pickle' in write-binary mode
with open("./extracted_landmarks.pickle", "wb") as f:
    # Save the 'dataset' and 'labels' as a dictionary using pickle for later use
    pickle.dump({"dataset": dataset, "labels": labels}, f)