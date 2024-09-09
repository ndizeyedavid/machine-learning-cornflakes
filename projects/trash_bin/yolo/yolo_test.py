# just skip all boring stuff and went with yolo8l which is indeed cool and accurate

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8l.pt') 

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Render the detections on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
