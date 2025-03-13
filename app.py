import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load the trained MoBiLSTM model
model = tf.keras.models.load_model("violence_model.keras")

# Define constants
frame_height, frame_width = 96, 96  # Same as used in training
sequence_length = 16
class_labels = ["NonViolence", "Violence"]

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Buffer to store frames for sequence processing
frame_buffer = deque(maxlen=sequence_length)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (frame_width, frame_height))
    normalized_frame = resized_frame.astype('float32') / 255.0

    # Append the frame to buffer
    frame_buffer.append(normalized_frame)

    # Predict only when we have enough frames
    if len(frame_buffer) == sequence_length:
        input_sequence = np.array(frame_buffer)  # Convert to NumPy array
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

        # Get model predictions
        prediction = model.predict(input_sequence)
        predicted_class = np.argmax(prediction)
        label = class_labels[predicted_class]

        # Display the prediction
        color = (0, 255, 0) if label == "NonViolence" else (0, 0, 255)
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the video feed
    cv2.imshow("Violence Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
