import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os  # For clearing the command prompt

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 
               23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 
               34: '8', 35: '9', 36: 'Space'}

# Variables for sentence construction and debounce
final_sentence = ""
last_prediction_time = 0  # To track the last prediction timestamp
debounce_interval = 3  # 3 seconds debounce interval
previous_character = ""  # To track the last detected character

# Function to resize the frame while maintaining aspect ratio
def resize_frame(frame, width):
    aspect_ratio = width / float(frame.shape[1])
    height = int(frame.shape[0] * aspect_ratio)
    return cv2.resize(frame, (width, height))

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a width of 640px
    frame = resize_frame(frame, 640)
    H, W, _ = frame.shape

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Collect landmarks data for prediction
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Get the bounding box for the hand region
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Check debounce interval before making a new prediction
        current_time = time.time()
        if current_time - last_prediction_time > debounce_interval:
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Add the predicted character to the sentence if it changes
            if predicted_character != previous_character:
                if predicted_character == "Space":
                    final_sentence += " "  # Add space
                else:
                    final_sentence += predicted_character
                previous_character = predicted_character
                last_prediction_time = current_time  # Update the last prediction time

            # Display the predicted character
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Draw bounding box around the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    # Display the constructed sentence on the frame
    cv2.putText(frame, f"Sentence: {final_sentence}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame with annotations
    cv2.imshow('frame', frame)

    # Clear the command prompt and print the last updated sentence
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear cmd/terminal screen
    print("Predicted Sentence: ", final_sentence)

    # Exit if 'q' key is pressed, or remove last character if 'b' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):  # 'b' key for backspace (delete last character)
        final_sentence = final_sentence[:-1]
        previous_character = ""  # Reset previous character to force re-checking prediction

# Release resources
cap.release()
cv2.destroyAllWindows()
