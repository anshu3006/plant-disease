"""Plant disease detection demo.  Run:  streamlit run app.py"""
import json
import pathlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).parent
MODEL_DIR = ROOT / "models"
DEFAULT_THRESHOLD = 0.6

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def pretty(class_name):
    plant, _, condition = class_name.partition("___")
    plant = plant.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip()
    return plant, condition


@st.cache_resource
def load_model():
    import keras  # imported here so the page shows quickly
    return keras.models.load_model(MODEL_DIR / "best_model.keras", compile=False)


@st.cache_data
def load_metadata():
    class_names = json.loads((MODEL_DIR / "class_names.json").read_text())
    info_path = MODEL_DIR / "model_info.json"
    model_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info = json.loads((ROOT / "disease_info.json").read_text(encoding="utf-8"))
    return class_names, model_info, {norm(k): v for k, v in info.items()}


st.title("🌿 Plant Disease Detection")
st.caption("Transfer learning (MobileNetV2 / ResNet50V2 / EfficientNetB0) fine-tuned on PlantVillage + PlantDoc")

if not (MODEL_DIR / "best_model.keras").exists():
    st.error("No trained model found. Run `train.ipynb` on Google Colab, then copy "
             "`best_model.keras`, `class_names.json` and `model_info.json` into the `models/` folder.")
    st.stop()

class_names, model_info, disease_info = load_metadata()
model = load_model()
img_size = model.input_shape[1]

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0,
                          float(model_info.get("confidence_threshold", DEFAULT_THRESHOLD)), 0.05)
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    st.markdown("**Tips for best results**\n- One leaf, close-up\n- Plain background (paper, table)\n"
                "- Good light, no blur")
    if model_info:
        st.caption(f"Model: {model_info.get('model', '?')}")

source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"]) if source == "Upload a photo" \
    else st.camera_input("Take a photo of a leaf")

if file is None:
    st.info("Upload or take a photo of a single leaf to get a diagnosis.")
    st.stop()

image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
arr = np.asarray(image.resize((img_size, img_size)), dtype=np.float32)

from gradcam import grad_cam, overlay  # noqa: E402  (imports TensorFlow)

with st.spinner("Analysing..."):
    cam, idx, probs = grad_cam(model, arr)

plant, condition = pretty(class_names[idx])
confidence = float(probs[idx])
info = disease_info.get(norm(class_names[idx]), {})

left, right = st.columns(2)
with left:
    st.image(image, caption="Your photo", width="stretch")
with right:
    if show_cam:
        st.image(overlay(arr.astype(np.uint8), cam), caption="Grad-CAM: red = where the model looked",
                 width="stretch")

if confidence < threshold:
    st.warning(f"Not sure ({confidence:.0%} confidence). Please upload a clearer close-up of a single "
               f"leaf on a plain background. Best guess: **{plant} – {condition}**.")
elif "healthy" in condition.lower():
    st.success(f"### {plant}: Healthy ✅  \nConfidence: {confidence:.1%}")
else:
    st.error(f"### {plant}: {condition}  \nConfidence: {confidence:.1%}")

if info and confidence >= threshold:
    st.markdown(f"**About:** {info['description']}")
    st.markdown(f"**What to do:** {info['treatment']}")

st.subheader("Top 3 predictions")
top = np.argsort(probs)[::-1][:3]
st.bar_chart(pd.DataFrame({"confidence": probs[top]},
                          index=[" – ".join(pretty(class_names[i])) for i in top]), horizontal=True)
