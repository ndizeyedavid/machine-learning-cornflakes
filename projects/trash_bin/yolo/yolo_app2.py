# this the giant step for dawudi, because it worked!!!!!!!!!!!!!!!!!!!! 😇😇😇😇 

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

# IoU threshold for deciding if the object is in the hand
iou_threshold = 0.3  # Adjust based on performance (0.3 is a good starting point)

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    # Unpack the bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the coordinates of the intersection box
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    # Calculate the area of the intersection box
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the areas of the individual boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the IoU (Intersection over Union)
    iou = intersection_area / (box1_area + box2_area - intersection_area) if (box1_area + box2_area - intersection_area) > 0 else 0
    
    return iou

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

                    # Object bounding box
                    object_box = [x1, y1, x2, y2]

                    # Calculate IoU between hand and object
                    iou = calculate_iou(hand_box, object_box)

                    if iou > iou_threshold:
                        # Print message indicating object is in hand
                        print(f"Holding {obj_label} with IoU: {iou:.2f}")
                        
                        # Optional: Draw label on the frame
                        cv2.putText(annotated_frame, f'Holding {obj_label}', (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        print(f"No object is in hand. IoU: {iou:.2f}")
    
    # Display the result
    cv2.imshow('Hand and Object Detection', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
