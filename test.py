from ultralytics import YOLO
import cv2
model = YOLO("waste beverage bottles.v4i.yolov8.pt")

model.predict(source="0", show=True, conf=0.5)
