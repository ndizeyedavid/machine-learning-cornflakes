
# very low accuracy but the start of detecting an object in my hands

import cv2
import torch
import numpy as np
import mediapipe as mp

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    hand_results = hands.process(rgb_frame)
    
    # YOLO object detection
    yolo_results = model(frame)

    # Render YOLO detections
    yolo_frame = np.squeeze(yolo_results.render())
    
    # Check for hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(yolo_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for hand
            h, w, _ = frame.shape
            hand_box = [
                min([lm.x for lm in hand_landmarks.landmark]) * w,  # xmin
                min([lm.y for lm in hand_landmarks.landmark]) * h,  # ymin
                max([lm.x for lm in hand_landmarks.landmark]) * w,  # xmax
                max([lm.y for lm in hand_landmarks.landmark]) * h   # ymax
            ]

            # Check if any object is within hand bounding box
            for obj in yolo_results.xyxy[0]:
                x1, y1, x2, y2, confidence, cls = obj
                obj_label = yolo_results.names[int(cls)]

                # Check if the object's bounding box overlaps with hand
                if (x1 >= hand_box[0] and x2 <= hand_box[2] and
                    y1 >= hand_box[1] and y2 <= hand_box[3]):
                    
                    # Print message indicating object is in hand
                    print(f"Holding {obj_label}")
                    
                    # Optional: draw a label on the frame
                    cv2.putText(yolo_frame, f'Holding {obj_label}', (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Object Detection with Hand Tracking', yolo_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
