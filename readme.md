# 🌒 Lunar Crater Detection – Craternauts Submission

This project detects **lunar craters and boulders** from surface imagery using an **ensemble of YOLOv8 and DETR** models. It supports end‑to‑end workflows including **training, inference, ensemble fusion**, and an **interactive GUI application** for visual predictions.

<img width="719" height="743" alt="Lunar crater detection results" src="https://github.com/user-attachments/assets/59d4c055-b0ee-440f-a01d-abacaa132381" />

---

## 🗂️ Project Structure

```text
craternauts_submission/
├── code_files/
│   ├── app.py              # GUI application for interactive predictions
│   ├── train.py            # YOLOv8 training script
│   ├── inference.py        # Inference pipeline
│   └── detr_ensemble.py    # YOLOv8 + DETR ensemble logic
├── trained_models/
│   ├── best.pt             # Trained YOLOv8 weights
│   └── detr_pt/            # DETR model directory
│       ├── config.json
│       └── model.safetensors
├── predicted_labels/       # YOLO-format prediction .txt files
└── test_images/             # Images for inference (optional)
```

---

## 🛠️ Installation

Install all required dependencies using pip:

```bash
pip install torch torchvision transformers ultralytics opencv-python matplotlib pillow tqdm
```

---

## 🧠 1. YOLOv8 Training

You can retrain the YOLOv8 model on your own lunar crater dataset.

### ▶️ Run Training

```bash
cd code_files
python train.py
```

The trained weights will be saved to:

```text
trained_models/best.pt
```

### ⚙️ Dataset Configuration (Important)

The `train.py` script dynamically generates a `dataset.yaml` file. You **must update the dataset paths** inside `train.py` to point to the **absolute locations** of your training and validation images.

```python
# Inside train.py
yaml_content = """
train: /absolute/path/to/train/images
val: /absolute/path/to/val/images

nc: 1
names: ['crater']
"""
```

Ensure that the image folders follow YOLO directory conventions and that corresponding label files exist.

---

## 🔍 2. Inference & Ensemble Detection

Inference uses an **ensemble strategy** combining predictions from **YOLOv8 and DETR** to improve robustness.

### 📁 Key Paths & Parameters

```python
import os
from PIL import Image
from tqdm import tqdm

from detr_ensemble import detect_ensemble

# Paths
TEST_IMAGE_DIR = 'test_images'      # Directory containing input images
OUTPUT_LABEL_DIR = 'predicted_labels'  # Output directory for YOLO-format labels
CONFIDENCE_THRESHOLD = 0.25         # Minimum confidence for detections
```

* YOLOv8 predictions are loaded from `best.pt`
* DETR predictions are loaded from `trained_models/detr_pt/`
* Final detections are merged and filtered using confidence thresholds

---

## 🧩 3. Ensemble Logic (YOLOv8 + DETR)

The ensemble process is implemented in:

```text
code_files/detr_ensemble.py
```

It performs:

* Independent detection using YOLOv8 and DETR
* Confidence-based filtering
* Merging overlapping detections
* Exporting results in YOLO label format

This improves crater detection consistency across varying crater sizes and illumination conditions.

---

## 🖥️ 4. GUI Application

The GUI allows users to:

* Upload lunar surface images
* Run ensemble inference
* Visualize detected craters interactively

To launch the app:

```bash
cd code_files
python app.py
```

---

## 📤 Output & Prediction File Format

Each prediction is saved as a `.txt` file following the **YOLO format**:

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
````

* All coordinates are **normalized to the range [0, 1]**
* Each file corresponds to **one input image**
* Prediction files are stored in: ```predicted_labels/```

---

## 🚨 Notes for Evaluators

- The ensemble logic in `detr_ensemble.py` automatically checks for **DETR availability**
- If DETR is missing or misconfigured, the system **falls back to YOLO-only detection**
- All detection functions return:
  - Bounding boxes
  - Confidence scores
  - Class IDs

---

## ✅ Quick Dependency Check

Run the following to verify all critical dependencies:

```
cd code_files  
python -c "import torch, transformers, ultralytics; print('✅ All dependencies loaded.')"
```

---

## 📞 Contact
For any doubts, issues, or errors, feel free to contact me at: **sameerchakrawarti513@gmail.com**
