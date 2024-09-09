import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


model = YOLO('yolov8l.pt')


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


CONFIDENCE_THRESHOLD = 0.5


iou_threshold = 0.3  


proximity_threshold = 50  

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = intersection_area / (box1_area + box2_area - intersection_area) if (box1_area + box2_area - intersection_area) > 0 else 0
    
    return iou

def is_near_hand(object_box, hand_box, proximity_threshold):
    """Checks if the object bounding box is near the hand bounding box based on proximity threshold."""
    x1_obj, y1_obj, x2_obj, y2_obj = object_box
    x1_hand, y1_hand, x2_hand, y2_hand = hand_box

    
    x1_hand_exp = x1_hand - proximity_threshold
    y1_hand_exp = y1_hand - proximity_threshold
    x2_hand_exp = x2_hand + proximity_threshold
    y2_hand_exp = y2_hand + proximity_threshold

    
    return (x1_obj < x2_hand_exp and x2_obj > x1_hand_exp and
            y1_obj < y2_hand_exp and y2_obj > y1_hand_exp)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    hand_results = hands.process(rgb_frame)
    
    
    results = model(frame)

    
    annotated_frame = np.squeeze(results[0].plot())
    
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            h, w, _ = frame.shape
            hand_box = [
                min([lm.x for lm in hand_landmarks.landmark]) * w,  
                min([lm.y for lm in hand_landmarks.landmark]) * h,  
                max([lm.x for lm in hand_landmarks.landmark]) * w,  
                max([lm.y for lm in hand_landmarks.landmark]) * h   
            ]

            
            for obj in results[0].boxes:
                x1, y1, x2, y2 = obj.xyxy[0]  
                confidence = obj.conf[0]  
                cls = obj.cls[0]  

                
                if confidence > CONFIDENCE_THRESHOLD:
                    obj_label = results[0].names[int(cls)]

                    
                    object_box = [x1, y1, x2, y2]

                    
                    if is_near_hand(object_box, hand_box, proximity_threshold):
                        
                        if (obj_label != 'person'):
                            print(f"Object {obj_label} is near the hand.")
                            cv2.putText(annotated_frame, f'{obj_label} is in Hand', (int(x1), int(y1)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        print(f"Object {obj_label} is not in hand.")
    
    
    cv2.imshow('Hand and Object Detection', annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
