import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# This is the local Imlpementation of the model for real-time hand gesture recognition.

# First, we need to load the model
model = tf.keras.models.load_model("best_model_key.keras") # The best model trained using the KeyPoints dataset

# Define the class names corresponding to the model's output
class_names = ['A', 'B', 'C','CH', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X']

# Now, we will set up MediaPipe for hand detection and keypoint extraction
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# We define the image size that the model expects
image_size = 512
cap = cv2.VideoCapture(0)

# We will use a loop to continuously capture frames from the webcam
while True:
    # To capture the video from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Create a black image to draw the hand landmarks
    black = np.zeros((height, width, 3), dtype=np.uint8)

    # If hands are detected, draw the landmarks on the black image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(black, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Uncomment to draw landmarks on the original frame

        
        # Resize the black image to the input size expected by the model
        black_resized = cv2.resize(black, (image_size, image_size))
        black_resized_norm = black_resized.astype(np.float32) / 255.0
        input_image = np.expand_dims(black_resized_norm, axis=0)

        # Uncomment the following lines if you want to use a model with the original frame with keypoints
        # processed = cv2.resize(frame, (image_size, image_size))
        #frame_with_keypoints = cv2.resize(frame, (image_size, image_size))
        #input_image = frame_with_keypoints.astype(np.float32) / 255.0
        #input_image = np.expand_dims(input_image, axis=0)


        # Display the processed frame with landmarks
        cv2.imshow("Processed Frame", black_resized)

        # Uncomment the following line if you want to see the frame with keypoints
        # cv2.imshow("Frame with Keypoints (Input to Model)", frame_with_keypoints)

        # Make predictions using the model
        pred = model.predict(input_image, verbose=0)
        pred_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        # Display the prediction on the original frame
        cv2.putText(frame, f"{pred_class} ({confidence:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    else:
        # If no hands are detected, display a message
        cv2.imshow("Processed Frame", black)

        # Uncomment the following line if you want to see the frame with keypoints
        #cv2.imshow("Frame with Keypoints (Input to Model)", frame)

    # Display the original frame with the prediction
    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()