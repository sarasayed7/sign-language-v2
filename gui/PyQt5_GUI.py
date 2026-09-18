import sys
import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent
from PyQt5.QtCore import QTimer, Qt
import os
import logging

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Suppress Mediapipe warnings


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load trained model
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']

        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils

        # Labels for predicted characters
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                            11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                            21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
                            31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: 'Space'}
        
    

        # UI Components
        self.initUI()

        # Variables
        self.cap = None  # Video Capture
        self.final_sentence = ""
        self.previous_character = ""
        self.last_prediction_time = 0
        self.debounce_interval = 3  # 0.5s debounce
        self.timer = QTimer(self)  # Timer for updating the video feed
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        """Initialize the GUI Layout"""
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Video Feed Label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Predicted Letter Label
        self.predicted_label = QLabel("Predicted Letter: ", self)
        self.predicted_label.setAlignment(Qt.AlignCenter)

        # Sentence Display
        self.sentence_display = QTextEdit(self)
        self.sentence_display.setReadOnly(True)

        # Buttons
        self.start_btn = QPushButton("Start Camera", self)
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("Stop Camera", self)
        self.stop_btn.clicked.connect(self.stop_camera)

        self.clear_btn = QPushButton("Clear", self)
        self.clear_btn.clicked.connect(self.clear_sentence)

        self.backspace_btn = QPushButton("Backspace", self)
        self.backspace_btn.clicked.connect(self.remove_last_character)

        self.save_btn = QPushButton("Save Sentence", self)
        self.save_btn.clicked.connect(self.save_text)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.predicted_label)
        layout.addWidget(self.sentence_display)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.backspace_btn)
        layout.addWidget(self.save_btn)

        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key_B:
            self.remove_last_character()

    def start_camera(self):
        """Start Webcam Feed"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # Update every 30ms

    def stop_camera(self):
        """Stop Webcam Feed"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()

    def clear_sentence(self):
        """Clear the entire sentence"""
        self.final_sentence = ""
        self.sentence_display.setText("")

    def remove_last_character(self):
        """Remove the last character (Backspace)"""
        self.final_sentence = self.final_sentence[:-1]
        self.sentence_display.setText(self.final_sentence)

    def save_text(self):
        """Save the sentence to a text file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sentence", "", "Text Files (*.txt)", options=options)
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.final_sentence)

    def update_frame(self):
        """Update the video frame and run the model"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape

        # Process hand landmarks
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Predict Sign Language Letter
            if len(data_aux) == 42:
                current_time = time.time()
                if current_time - self.last_prediction_time > self.debounce_interval:
                    prediction = self.model.predict([np.asarray(data_aux)])
                    predicted_character = self.labels_dict[int(prediction[0])]

                    if predicted_character != self.previous_character:
                        if predicted_character == "Space":
                            self.final_sentence += " "
                        else:
                            self.final_sentence += predicted_character
                        self.previous_character = predicted_character
                        self.last_prediction_time = current_time

                    # Update UI
                    self.predicted_label.setText(f"Predicted Letter: {predicted_character}")
                    self.sentence_display.setText(self.final_sentence)

        # Convert frame to display in PyQt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
