import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

# --------------------------------------
# Setup
# --------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔁 CHANGE THIS PATH if you rename best.pt
yolo = YOLO("trained_models/best.pt")

# Try loading DETR
try:
    # 🔁 CHANGE THIS PATH if folder is renamed
    processor = DetrImageProcessor.from_pretrained("trained_models/detr_pt")
    detr = DetrForObjectDetection.from_pretrained("trained_models/detr_pt").to(device)
    detr_available = True
except Exception as e:
    print("⚠️ DETR model could not be loaded. Proceeding with YOLO-only fallback.")
    print("Reason:", e)
    detr_available = False

# --------------------------------------
# Inference Functions
# --------------------------------------

def detect_detr(image_pil, threshold=0.5):
    if not detr_available:
        return [], [], []

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    outputs = detr(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)

    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    classes = results["labels"].cpu().numpy()
    return boxes, scores, classes


def detect_yolo(image_pil, conf=0.25):
    results = yolo.predict(image_pil, conf=conf)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, classes


def ensemble_boxes(boxes1, scores1, classes1, boxes2, scores2, classes2, iou_thresh=0.5):
    all_boxes = np.vstack((boxes1, boxes2))
    all_scores = np.concatenate((scores1, scores2))
    all_classes = np.concatenate((classes1, classes2))

    idxs = torch.ops.torchvision.nms(
        torch.tensor(all_boxes, dtype=torch.float32),
        torch.tensor(all_scores, dtype=torch.float32),
        iou_thresh
    )
    selected = idxs.cpu().numpy()
    return all_boxes[selected], all_scores[selected], all_classes[selected]


def detect_ensemble(image_pil, conf=0.25, detr_thresh=0.5, iou_thresh=0.5):
    y_boxes, y_scores, y_classes = detect_yolo(image_pil, conf)

    if detr_available:
        d_boxes, d_scores, d_classes = detect_detr(image_pil, threshold=detr_thresh)
    else:
        d_boxes, d_scores, d_classes = [], [], []

    if len(d_boxes) == 0:
        return y_boxes, y_scores, y_classes

    boxes, scores, classes = ensemble_boxes(
        y_boxes, y_scores, y_classes,
        d_boxes, d_scores, d_classes,
        iou_thresh
    )
    return boxes, scores, classes

# --------------------------------------
# Visualization (optional)
# --------------------------------------

def show_ensemble(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    boxes, scores, classes = detect_ensemble(img_pil)

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("YOLOv8 + DETR Ensemble Detection")
    plt.show()

# --------------------------------------
# Test (Optional)
# --------------------------------------

if __name__ == "__main__":
    # 🔁 CHANGE THIS PATH to any valid test image
    test_image_path = "sample_test_images/EXAMPLE.png"
    show_ensemble(test_image_path)
