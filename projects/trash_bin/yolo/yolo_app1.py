# detects my hand but harder to detect if i'm holding something

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model (large version)
model = YOLO('yolov8l.pt')

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Confidence threshold for YOLOv8 object detection
CONFIDENCE_THRESHOLD = 0.5

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB for MediaPipe hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    hand_results = hands.process(rgb_frame)
    
    # Object detection with YOLOv8
    results = model(frame)

    # Render YOLOv8 detections
    annotated_frame = np.squeeze(results[0].plot())
    
    # Check for hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for hand
            h, w, _ = frame.shape
            hand_box = [
                min([lm.x for lm in hand_landmarks.landmark]) * w,  # xmin
                min([lm.y for lm in hand_landmarks.landmark]) * h,  # ymin
                max([lm.x for lm in hand_landmarks.landmark]) * w,  # xmax
                max([lm.y for lm in hand_landmarks.landmark]) * h   # ymax
            ]

            # Check if any object is within the hand bounding box
            for obj in results[0].boxes:
                # Extract the bounding box coordinates and confidence
                x1, y1, x2, y2 = obj.xyxy[0]  # Bounding box
                confidence = obj.conf[0]  # Confidence score
                cls = obj.cls[0]  # Class index

                # Only consider objects with confidence greater than the threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    obj_label = results[0].names[int(cls)]

                    # Check if the object's bounding box overlaps with hand
                    if (x1 >= hand_box[0] and x2 <= hand_box[2] and
                        y1 >= hand_box[1] and y2 <= hand_box[3]):
                        
                        # Print message indicating object is in hand
                        print(f"Holding {obj_label}")
                        
                        # Optional: Draw label on the frame
                        cv2.putText(annotated_frame, f'Holding {obj_label}', (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        print("No object is in hand")
    
    # Display the result
    cv2.imshow('Hand and Object Detection', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
