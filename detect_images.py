import cv2
import numpy as np
import os

# ✅ Load YOLOv4-Tiny model using your existing weights and config
weights_path = "yolo-tiny/yolov4-tiny.weights"
config_path = "yolo-tiny/yolov4-tiny.cfg"
labels_path = "yolo-tiny/coco.names"  # Change to your custom class names if needed

# Load class labels
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# ✅ Input & Output Paths
input_folder = "yolov4-Tiny-Pytorch/dataset/images/test/"  # Images for detection
output_folder = "detected_images/"  # Output folder for detected images

# ✅ Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# ✅ Process each image in the test dataset
for img_file in os.listdir(input_folder):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)

        # ✅ Load image
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # ✅ Convert image to blob for YOLO processing
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # ✅ Run forward pass to get detections
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # ✅ Process detections
        class_ids, confidences, boxes = [], [], []
        for detection in outs:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    center_x, center_y, w, h = (obj[0:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # ✅ Apply Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # ✅ Draw bounding boxes on the image
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ✅ Save the detected image
        cv2.imwrite(output_path, image)
        print(f"✅ Detection complete: {output_path}")

print("🎯 All images processed! Check 'detected_images/' for results.")
