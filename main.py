from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")

# Train the model
train_results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=30,
    imgsz=640
)
