import os
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
