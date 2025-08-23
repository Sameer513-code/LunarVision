import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Page setup
st.set_page_config(page_title="Lunar Crater Detection", layout="centered")
st.title("🌕 Lunar Crater Detection UI")
st.markdown("Conf threshold filters predictions below this confidence to remove false positives.")

import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # get path to app.py
    model_path = os.path.join(base_dir, "..", "trained_models", "best.pt")

    if not os.path.exists(model_path):
        st.error(f"❌ YOLOv8 model not found at:\n`{model_path}`\n\nPlease check if `best.pt` exists.")
        st.stop()

    model = YOLO(model_path)
    model.fuse()
    return model


model = load_model()

# Confidence slider
conf_thresh = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# File upload
uploaded_file = st.file_uploader("Upload a Lunar Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(input_image)
    img_h, img_w = image_np.shape[:2]

    # Model prediction
    results = model.predict(source=image_np, save=False, conf=conf_thresh, device='cpu')[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()
    crater_count = len(boxes)

    # Drawing + crater info
    boxed_img = image_np.copy()
    crater_data = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        diameter = (x2 - x1 + y2 - y1) // 2

        crater_data.append({
            "x": cx,
            "y": cy,
            "diameter": diameter,
            "score": round(score, 2)
        })

        # Draw bounding box
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw multiline label
        label_lines = [
            f"{int(score*100)}%",        # Confidence
        ]
        for i, line in enumerate(label_lines):
            cv2.putText(boxed_img, line, (x1, y1 - 15 - 15*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display image
    st.image(boxed_img, caption=f"Detected Craters: {crater_count}", use_column_width=True)
    st.metric("🕳️ Crater Count", crater_count)

    # --- Crater Density Heatmap ---
    grid_size = 10
    density_map = np.zeros((grid_size, grid_size))

    for crater in crater_data:
        gx = min(grid_size - 1, int(crater["x"] / img_w * grid_size))
        gy = min(grid_size - 1, int(crater["y"] / img_h * grid_size))
        density_map[gy, gx] += 1

    # Normalize density map
    density_map /= np.max(density_map) if np.max(density_map) > 0 else 1

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_title("🚀 Crater Density Map (10x10 Grid)")
    heatmap = ax1.imshow(density_map, cmap="YlOrRd", origin="upper")
    plt.colorbar(heatmap, ax=ax1, label="Normalized Crater Count")
    st.pyplot(fig1)

    # --- Interactive Crater Info ---
    st.subheader(":mag: Crater Info Viewer (Hover)")
    crater_df = pd.DataFrame(crater_data)

    fig2 = px.scatter(crater_df, x="x", y="y", size="diameter", color="score",
                      hover_data=["diameter", "score"],
                      title="Crater Info about its Coordinates (x,y), Diameter and Circularity Score.",
                      labels={"x": "X-Center", "y": "Y-Center"})
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Crater Age Estimation (Degradation) ---
    st.subheader(":satellite: Crater Age Estimation")
    for crater in crater_data:
        crater["degradation"] = np.clip(crater["diameter"] / max(img_w, img_h), 0, 1)

    age_plot = px.histogram(crater_data, x="degradation", nbins=10,
                            labels={"degradation": "Degradation Score"},
                            title="Crater Age Proxy (0 = Fresh, 1 = Degraded)")
    st.plotly_chart(age_plot, use_container_width=True)
