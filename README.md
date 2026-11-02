# 🌿 Plant Disease Detection using Transfer Learning

Detects 38 plant leaf diseases (14 crops) from a photo. ImageNet-pretrained CNNs are fine-tuned on
**PlantVillage**, then adapted to real field photos from **PlantDoc**. Grad-CAM heatmaps show which
part of the leaf the model used.

## Project files

| File | What it is |
|---|---|
| `train.ipynb` | Colab notebook: download data, train and compare all models, Grad-CAM, save the final model |
| `app.py` | Streamlit web app: upload or take a leaf photo, get the diagnosis + heatmap |
| `gradcam.py` | Grad-CAM code used by the app |
| `disease_info.json` | Description and treatment for each of the 38 classes |
| `models/` | Put the trained model files here |
| `notebook_src/build_notebook.py` | Source that generates `train.ipynb` (edit here, then rebuild) |
| `REPORT_OUTLINE.md` | Report structure with what to write in each section |

## Step 1: train on Google Colab (about 1.5-2.5 hours)

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `train.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**. Allow Google Drive access when asked.
4. Everything is saved to `MyDrive/plant_disease/` as it finishes. If Colab disconnects,
   reconnect and run again; finished models are already in Drive.
5. Download these three files into this project's `models/` folder:
   - `best_model.keras`
   - `class_names.json`
   - `model_info.json`
6. Also download the `.png` and `.csv` files; they are your report's figures and tables.

> If Colab says "GPU limit reached", wait a day or lower the epochs in section 2 of the notebook.

## Step 2: run the app on your laptop

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at <http://localhost:8501>. Upload a leaf photo or use the camera.

If loading the model fails with a version error, install the TensorFlow version printed at the end
of the notebook (it is also in `models/model_info.json`):

```bash
pip install tensorflow==<version from model_info.json>
```

**Demo tips:** one leaf, close-up, plain background (white paper), good light. Keep a few
PlantVillage test images as a backup.

## Step 3 (optional): put it online for your resume

Free hosting on **Hugging Face Spaces**:
1. Create an account at <https://huggingface.co> → **New Space** → SDK: **Streamlit**.
2. Upload `app.py`, `gradcam.py`, `disease_info.json`, `requirements.txt` and the `models/` folder.
3. The Space builds and gives you a public link.

## Method summary

- **Data:** PlantVillage, 54,305 images, 38 classes, split 70/15/15 (stratified).
  PlantDoc field photos, mapped to the PlantVillage classes they share (28 PlantDoc folders).
- **Models:** custom CNN (from scratch), MobileNetV2, ResNet50V2, EfficientNetB0.
- **Transfer learning:** stage 1 trains only the new classifier head (backbone frozen, lr 1e-3);
  stage 2 unfreezes the top 30% of the backbone (lr 1e-5, BatchNorm kept frozen).
- **Augmentation:** random flip, rotation, zoom, contrast, brightness. Ablation trains MobileNetV2 without it.
- **Domain adaptation:** best model fine-tuned on PlantDoc + a 6,000-image PlantVillage sample.
- **Metrics:** accuracy, macro F1, confusion matrix, per-class report, size, inference time,
  accuracy on real field photos.
- **Explainability:** Grad-CAM on the last convolutional layer.

## Known limitation

Models trained only on PlantVillage (lab photos, plain background) lose a lot of accuracy on real
field photos. The notebook measures this drop and reduces it with augmentation and PlantDoc
fine-tuning, but field accuracy stays well below lab accuracy.
