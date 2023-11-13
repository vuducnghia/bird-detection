from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")

results = model.predict(save=True, source="000487.png", imgsz=480)
print(results)