
# 🌒 Lunar Crater Detection – Craternauts Submission

This project uses an ensemble of YOLOv8 and DETR to detect craters and boulders on lunar surface images. It includes model training, inference generation, and a full GUI app for interactive predictions.

---

## 🛠 Folder Structure

```

craternauts\_submission/
├── code\_files/
│   ├── app.py
│   ├── train.py
│   ├── inference.py
│   └── detr\_ensemble.py
├── trained\_models/
│   ├── best.pt            # Trained YOLOv8 model
│   └── detr\_pt/           # DETR model folder (must contain config.json & model.safetensors)
├── predicted\_labels/      # Generated YOLO-format prediction text files
└── test\_images/           # Folder with test images (optional)

````

---

## 🔧 Install dependencies

Use the following command in PowerShell or terminal:
```bash
"C:\Users\schak\AppData\Local\Programs\Python\Python310\python.exe" -m pip install torch torchvision transformers ultralytics opencv-python matplotlib pillow
````

## 🧠 1. Training the YOLOv8 Model

To retrain the model using your own dataset:

```bash
cd code_files
python train.py
````

* The model will be saved as `trained_models/best.pt`.

### ⚙️ Required Update in `train.py`:

The `train.py` script auto-generates a `dataset.yaml` file before training.

Update the following lines to match the **absolute path** to your dataset:

```python
# Inside train.py
yaml_content = """
train: /path/to/your/train/images
val: /path/to/your/val/images

nc: 1
names: ['crater']
"""
```

Change the `/path/to/your/...` with actual locations of your `images/` folder from the training and validation sets.

deliveries.csvV
## 🔍 2.import os
from PIL import Image
from tqdm import tqdm

from detr_ensemble import detect_ensemble

# Paths
TEST_IMAGE_DIR = 'test_images'  # change if your test images are elsewhere
OUTPUT_LABEL_DIR = 'Predicted Labels'
CONFIDENCE_THRESHOLD = 0.25

# Ensure output directory exists
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# List all image files (you can filter for .png/.jpg if needed)
image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop over test images
for image_name in tqdm(image_files, desc="Running inference"):
    image_path = os.path.join(TEST_IMAGE_DIR, image_name)
    image = Image.open(image_path).convert("RGB")

    # Run ensemble detection
    boxes, scores, classes = detect_ensemble(image, conf=CONFIDENCE_THRESHOLD)

    # Prepare output file path
    txt_filename = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(OUTPUT_LABEL_DIR, txt_filename)

    # Write predictions to file
    with open(label_path, "w") as f:
        for box, score, cls in zip(boxes, scores, classes):
            if score < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.4f}\n")

print(f"\nDone. Predictions saved in '{OUTPUT_LABEL_DIR}/'.")
V Running Inference (Ensemble of YOLO + DETR)

Run inference over all test images:

```bash
cd code_files
python inference.py
```

* Output prediction files will be saved in `predicted_labels/`. Although Predictions have been made on test images and saved.

### ⚙️ Required Updates:

#### In `inference.py`:

```python
TEST_DIR = "path/to/test/images"
OUTPUT_DIR = "predicted_labels"
```

#### In `detr_ensemble.py`:

```python
yolo = YOLO("trained_models/best.pt")
processor = DetrImageProcessor.from_pretrained("trained_models/detr_pt")
detr = DetrForObjectDetection.from_pretrained("trained_models/detr_pt")
```

---

## 🖼️ 3. GUI App (Optional)

To run the Streamlit GUI:

```bash
cd code_files
streamlit run app.py
```

### ⚙️ Required Updates in `app.py`:

```python
MODEL_PATH = "trained_models/best.pt"
DETR_PATH = "trained_models/detr_pt"
```

* Upload an image and get bounding box predictions live.

---

## 📂 Format of Prediction Files

Each `.txt` file follows YOLO format:

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Coordinates are normalized \[0–1].

---

## 🚨 Notes for Evaluators

* The ensemble logic (in `detr_ensemble.py`) checks if DETR is available.
* If not, it falls back to YOLO-only detection.
* All detection functions return: `boxes`, `scores`, and `class IDs`.

---

## ✅ Quick Dependency Check

To verify everything is installed correctly:

```bash
cd code_files
"C:\Users\schak\AppData\Local\Programs\Python\Python310\python.exe" -c "import torch, transformers, ultralytics; print('✅ All dependencies loaded.')"
```

---

## 📞 Contact

For any doubts, contact:
**Sameer Chakrawarti**
Email: [cs24bt023@iitdh.ac.in](mailto:cs24bt023@iitdh.ac.in)
