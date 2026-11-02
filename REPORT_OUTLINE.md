# Report Outline: Plant Disease Detection using Transfer Learning

Follows the six sections required in the activity sheet. Fill the numbers from `results.csv` and
`real_world_results.csv` after training, and use the `.png` files as figures.

## 1. Abstract (150-200 words)
- Problem: crop diseases cause large yield losses; experts are not always available to farmers.
- Approach: transfer learning with MobileNetV2, ResNet50V2 and EfficientNetB0 fine-tuned on
  PlantVillage (54K images, 38 classes), compared against a CNN trained from scratch.
- Real-world check: tested on PlantDoc field photos, then domain-adapted.
- Results: best test accuracy __%, field-photo accuracy improved from __% to __%.
- Output: Streamlit web app with Grad-CAM explanations.

## 2. Literature Review
Summarise 5-6 papers in a table (Author, Year, Method, Dataset, Accuracy, Limitation):
- Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*: AlexNet/GoogLeNet on PlantVillage, ~99%.
- Hughes & Salathé (2015), the PlantVillage dataset paper.
- Ferentinos (2018), *Deep learning models for plant disease detection and diagnosis*: VGG, 99.5%.
- Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*: shows the drop on field images.
- Sandler et al. (2018) MobileNetV2; He et al. (2016) ResNet; Tan & Le (2019) EfficientNet.
- Selvaraju et al. (2017), *Grad-CAM*.

Research gap: most work reports lab-image accuracy only; few measure performance on real field photos.

## 3. Description of the Algorithm
- CNN basics: convolution, pooling, activation, softmax.
- Transfer learning: why ImageNet features (edges, textures) help with a small domain dataset.
- The three backbones: depthwise separable convolutions (MobileNetV2), residual connections
  (ResNet50V2), compound scaling (EfficientNet).
- Two-stage fine-tuning and why BatchNorm stays frozen.
- Data augmentation and domain adaptation.
- Grad-CAM: gradient of the class score with respect to the last conv feature map.
- Figure: block diagram (Image → Augmentation → Backbone → GAP → Dropout → Dense(38) → Softmax).

## 4. Algorithm (step by step)
1. Load PlantVillage images, split 70/15/15 stratified.
2. Resize to 224×224; apply random flip, rotation, zoom, contrast, brightness (training only).
3. Load backbone with ImageNet weights, remove top, freeze it.
4. Add GlobalAveragePooling → Dropout(0.3) → Dense(38, softmax).
5. Train head with Adam (lr 1e-3), sparse categorical cross-entropy.
6. Unfreeze top 30% of backbone (BatchNorm frozen), train with lr 1e-5.
7. Keep the best epoch by validation accuracy (early stopping).
8. Evaluate on test set: accuracy, macro F1, confusion matrix.
9. Evaluate on PlantDoc field photos; fine-tune best model on PlantDoc + PlantVillage sample.
10. Generate Grad-CAM heatmaps; deploy the final model in a Streamlit app.

## 5. Program Code
- Key parts of `train.ipynb`: data pipeline, `build_transfer`, `train_transfer`, `grad_cam`.
- `app.py` (prediction + threshold + Grad-CAM).
- Put the full code in an appendix or link the GitHub repo.

## 6. Results, Analysis and Screenshots
Tables:
- Model comparison (from `results.csv`): accuracy, macro F1, params, size, ms/image, training time.
- Lab vs field accuracy (from `real_world_results.csv`).

Figures:
- `class_distribution.png`, `sample_images.png`
- `training_curves.png`, `model_comparison.png`
- `confusion_matrix.png` (discuss which diseases get confused, e.g. early vs late blight)
- `gradcam_plantvillage.png`, `gradcam_plantdoc.png`
- `real_world_comparison.png`
- App screenshots: healthy leaf, diseased leaf, low-confidence warning.

Points to discuss:
- Transfer learning vs from-scratch: accuracy and training time.
- Accuracy vs size/speed trade-off (MobileNetV2 suits phones).
- Effect of augmentation (ablation), especially on field photos.
- Why ~99% on PlantVillage is not the full story: plain backgrounds, the field-photo drop.
- Grad-CAM: does the model look at lesions or background?
- Limitations and future work: more field data, leaf segmentation, mobile app, more crops.

## 7. Conclusion and References
