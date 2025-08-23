# train.py

import os
from ultralytics import YOLO

# Step 1: Create dataset YAML file (if it doesn't already exist)
yaml_content = """
train: /path/to/your/train/images
val: /path/to/your/val/images

nc: 1
names: ['crater']
"""
yaml_path = "dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

# Step 2: Load model
model = YOLO("yolov8n.pt")

# Step 3: Train the model
model.train(
    data=yaml_path,
    epochs=10,
    imgsz=320,
    batch=16,
    project="runs",
    name="crater_yolov8",
    exist_ok=True  # Overwrites previous runs/crater_yolov8 if exists
)

# Optional: Save trained model explicitly
model_path = "crater_yolov8_best.pt"
model.save(model_path)
print(f"Model saved at {model_path}")
