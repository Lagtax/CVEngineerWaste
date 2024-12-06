import torch
from ultralytics import YOLO
import cv2

# Check and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the model
model = YOLO("waste beverage bottles.v4i.yolov8.pt")

# Set the device after model loading
model.to(device)


# Prediction parameters for optimal performance
def predict_bottles():
    results = model.predict(
        source="0",  # Camera source
        show=True,  # Display video
        conf=0.3,  # Lower confidence threshold
        iou=0.5,  # Intersection over Union threshold
        imgsz=320,  # Reduced resolution
        stream=True,  # Continuous streaming
        verbose=False,  # Suppress logging
        max_det=3  # Limit detections per frame
    )

    # Iterate through the results stream
    for result in results:
        # Get the boxes, classes, and confidence scores
        boxes = result.boxes

        # Only print if there are detected objects
        if len(boxes) > 0:
            # Print labels and confidence for detected objects
            for box in boxes:
                # Get the class label and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # Print to terminal
                print(f"Detected: {label} (Confidence: {conf:.2f})")


# Run the prediction
if __name__ == "__main__":
    predict_bottles()