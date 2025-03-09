from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)