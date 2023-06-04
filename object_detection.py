# object_detection.py

import cv2
import numpy as np
import os

# Load YOLOv5 model
model_path = os.path.join(os.getcwd(), 'yolov5', 'yolov5s.pt')
net = cv2.dnn.readNet(model_path)

# Define classes
classes = ['object', 'npc', 'item']

# Define colors for bounding boxes
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

def detect_objects(image):
    """
    Detects objects, NPCs, and items in the given image using YOLOv5.
    Returns a list of dictionaries, where each dictionary contains the following keys:
    - 'class': the class of the detected object (e.g., 'object', 'npc', 'item')
    - 'confidence': the confidence score of the detection
    - 'bbox': the bounding box coordinates of the detection (x, y, w, h)
    """
    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)

    # Set input and output layers
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass
    outputs = net.forward(output_layers)

    # Postprocess detections
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                detections.append({'class': classes[class_id], 'confidence': confidence, 'bbox': (x, y, width, height)})

    # Draw bounding boxes
    for detection in detections:
        class_id = classes.index(detection['class'])
        color = colors[class_id]
        x, y, w, h = detection['bbox']
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return detections