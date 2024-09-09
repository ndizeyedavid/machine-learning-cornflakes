# this detects all objects and return there actual name with the accuracy
import cv2
import torch
import numpy as np 

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # YOLO model inference
    results = model(frame)

    # Display the results
    cv2.imshow('Object Detection', np.squeeze(results.render()))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
