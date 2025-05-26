import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the COCO-trained YOLOv4-Tiny model (using .cfg and .weights)
coco_cfg_path = "yolov4-tiny.cfg"  # Path to YOLOv4-Tiny .cfg file
coco_weights_path = "yolov4-tiny.weights"  # Path to YOLOv4-Tiny .weights file
coco_net = cv2.dnn.readNetFromDarknet(coco_cfg_path, coco_weights_path)
coco_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
coco_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the custom-trained YOLO model (using .pt)
custom_model = YOLO("runs/detect/yolov4-tiny-custom6/weights/best.pt")  # Path to your custom-trained .pt file

# COCO class names
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    """ Compute Intersection over Union (IoU) between two bounding boxes. """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    union = area_box1 + area_box2 - intersection

    # Compute IoU
    return intersection / union if union > 0 else 0

# Function to run inference with the COCO model
def detect_with_coco(image):
    (H, W) = image.shape[:2]
    ln = coco_net.getLayerNames()
    unconnected_out_layers = coco_net.getUnconnectedOutLayers()
    
    if unconnected_out_layers.ndim == 1:
        ln = [ln[i - 1] for i in unconnected_out_layers]
    else:
        ln = [ln[i[0] - 1] for i in unconnected_out_layers]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    coco_net.setInput(blob)
    layer_outputs = coco_net.forward(ln)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.25:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Start FPS timer
    start_time = time.time()

    # Run inference with the COCO model
    coco_boxes, coco_confidences, coco_class_ids = detect_with_coco(frame)

    # Run inference with the custom-trained model
    custom_results = custom_model(frame)
    custom_boxes = custom_results[0].boxes.xyxy.cpu().numpy() if len(custom_results[0].boxes) > 0 else []
    custom_classes = custom_results[0].boxes.cls.cpu().numpy() if len(custom_results[0].boxes) > 0 else []
    custom_confidences = custom_results[0].boxes.conf.cpu().numpy() if len(custom_results[0].boxes) > 0 else []

    # Stop FPS timer and calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)  # FPS formula

    # Apply NMS for COCO detections
    indices = cv2.dnn.NMSBoxes(coco_boxes, coco_confidences, 0.25, 0.4)
    filtered_coco_boxes, filtered_coco_confidences, filtered_coco_class_ids = [], [], []
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            overlap = False
            for custom_box in custom_boxes:
                if compute_iou(coco_boxes[i], custom_box) > 0.3:  # Remove overlapping detections
                    overlap = True
                    break
            if not overlap:
                filtered_coco_boxes.append(coco_boxes[i])
                filtered_coco_confidences.append(coco_confidences[i])
                filtered_coco_class_ids.append(coco_class_ids[i])

    # Draw FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw custom detections labeled as "DDC"
    for box, conf in zip(custom_boxes, custom_confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"DDC {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw filtered COCO detections
    for box, conf, class_id in zip(filtered_coco_boxes, filtered_coco_confidences, filtered_coco_class_ids):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{COCO_CLASS_NAMES[class_id]} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to quit the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
