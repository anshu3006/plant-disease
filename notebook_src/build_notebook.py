"""Builds train.ipynb from the cells below. Edit cells here, then run:
    python notebook_src/build_notebook.py
"""
import pathlib
import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip()))


md("""
# Plant Disease Detection using Transfer Learning

**How to run (Google Colab):**
1. `Runtime → Change runtime type → T4 GPU`
2. `Runtime → Run all`
3. Allow Google Drive access when asked. Every model, plot and table is saved to
   `MyDrive/plant_disease/` so nothing is lost if Colab disconnects.
4. When finished, download `best_model.keras` and `class_names.json` from that
   folder and put them in the project's `models/` folder.

**What this notebook does**
1. Downloads **PlantVillage** (54K lab images, 38 classes) and **PlantDoc** (real field photos).
2. Trains a **custom CNN from scratch** (baseline).
3. Fine-tunes **MobileNetV2, ResNet50V2, EfficientNetB0** (ImageNet-pretrained) — transfer learning.
4. Ablation: MobileNetV2 **without** data augmentation.
5. Compares accuracy, F1, model size and speed.
6. Tests every model on **real field photos** (PlantDoc) and then adapts the best one to them.
7. Grad-CAM heatmaps showing where the model looks.
""")

md("## 1. Setup")
code("""
import os, json, time, random, pathlib, subprocess, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
GPU = tf.config.list_physical_devices("GPU")
print("GPU:", GPU if GPU else "NONE - set Runtime > Change runtime type > T4 GPU")

try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = pathlib.Path("/content/drive/MyDrive/plant_disease")
    DATA_DIR = pathlib.Path("/content/data")
except ImportError:  # running outside Colab
    OUT = pathlib.Path("outputs")
    DATA_DIR = pathlib.Path("data")
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
keras.utils.set_random_seed(SEED)
if GPU:
    keras.mixed_precision.set_global_policy("mixed_float16")  # ~2x faster on T4
AUTOTUNE = tf.data.AUTOTUNE
""")

md("## 2. Configuration\nLower the epochs if Colab keeps disconnecting.")
code("""
IMG_SIZE = 224
BATCH = 64
CNN_EPOCHS = 8          # custom CNN from scratch
HEAD_EPOCHS = 3         # transfer learning stage 1: train only the new head
FINETUNE_EPOCHS = 3     # stage 2: unfreeze top layers of the backbone
PLANTDOC_EPOCHS = 8     # domain adaptation on real field photos
RUN_ABLATION = True     # extra MobileNetV2 run without augmentation
CONFIDENCE_THRESHOLD = 0.6
""")

md("## 3. Download datasets")
code("""
def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

PV_ROOT = DATA_DIR / "PlantVillage-Dataset" / "raw" / "color"
PD_ROOT = DATA_DIR / "PlantDoc-Dataset"

if not PV_ROOT.exists():
    try:
        run(["git", "clone", "--depth", "1",
             "https://github.com/spMohanty/PlantVillage-Dataset",
             str(DATA_DIR / "PlantVillage-Dataset")])
    except Exception as e:
        print("GitHub download failed, trying Kaggle mirror:", e)
        run(["pip", "install", "-q", "kagglehub"])
        import kagglehub
        kaggle_dir = pathlib.Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
        PV_ROOT = next(p for p in kaggle_dir.rglob("color") if p.is_dir())

if not PD_ROOT.exists():
    run(["git", "clone", "--depth", "1",
         "https://github.com/pratikkayal/PlantDoc-Dataset", str(PD_ROOT)])

print("PlantVillage:", PV_ROOT)
print("PlantDoc:", PD_ROOT)
""")

md("## 4. Load PlantVillage and split 70 / 15 / 15")
code("""
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(folder):
    return sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)

class_names = sorted(d.name for d in PV_ROOT.iterdir() if d.is_dir())
NUM_CLASSES = len(class_names)

paths, labels = [], []
for i, c in enumerate(class_names):
    files = list_images(PV_ROOT / c)
    paths += [str(f) for f in files]
    labels += [i] * len(files)
paths, labels = np.array(paths), np.array(labels)

tr_p, tmp_p, tr_y, tmp_y = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
va_p, te_p, va_y, te_y = train_test_split(tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED)
print(f"{NUM_CLASSES} classes | {len(paths)} images | train {len(tr_p)}  val {len(va_p)}  test {len(te_p)}")

with open(OUT / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=1)
""")

code("""
def load_image(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # float32, 0-255
    return img, label

def make_ds(p, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        ds = ds.shuffle(len(p), seed=SEED)
    return ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds(tr_p, tr_y, shuffle=True)
val_ds = make_ds(va_p, va_y)
test_ds = make_ds(te_p, te_y)
""")

md("### Look at the data")
code("""
counts = pd.Series(labels).map(lambda i: class_names[i]).value_counts()
plt.figure(figsize=(12, 8))
counts.sort_values().plot.barh()
plt.title("Images per class (PlantVillage)")
plt.tight_layout(); plt.savefig(OUT / "class_distribution.png", dpi=120); plt.show()

imgs, ys = next(iter(make_ds(tr_p[:16], tr_y[:16])))
plt.figure(figsize=(12, 12))
for k in range(16):
    plt.subplot(4, 4, k + 1)
    plt.imshow(imgs[k].numpy().astype("uint8"))
    plt.title(class_names[ys[k]].replace("___", "\\n"), fontsize=8); plt.axis("off")
plt.tight_layout(); plt.savefig(OUT / "sample_images.png", dpi=120); plt.show()
""")

md("""
## 5. Models
Every model takes raw 0-255 pixels, so the app needs no extra preprocessing.
Augmentation layers are only active during training.
""")
code("""
def make_augment():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augment")

def head(x):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

def build_cnn():
    \"\"\"Custom CNN trained from scratch (baseline).\"\"\"
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = make_augment()(inputs)
    x = layers.Rescaling(1 / 255.0)(x)
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    return keras.Model(inputs, head(x), name="CustomCNN")

# backbone constructor, input scaling it expects (EfficientNet rescales internally)
BACKBONES = {
    "MobileNetV2": (keras.applications.MobileNetV2, lambda: layers.Rescaling(1 / 127.5, offset=-1)),
    "ResNet50V2": (keras.applications.ResNet50V2, lambda: layers.Rescaling(1 / 127.5, offset=-1)),
    "EfficientNetB0": (keras.applications.EfficientNetB0, None),
}

def build_transfer(name, augment=True):
    ctor, scaling = BACKBONES[name]
    base = ctor(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = make_augment()(inputs) if augment else inputs
    if scaling:
        x = scaling()(x)
    x = base(x, training=False)  # keep BatchNorm in inference mode
    return keras.Model(inputs, head(x), name=name + ("" if augment else "_noaug")), base

def unfreeze_top(base, fraction=0.3):
    base.trainable = True
    cut = int(len(base.layers) * (1 - fraction))
    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
""")

md("## 6. Training helpers")
code("""
def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def fit(model, epochs, tr=None, va=None):
    path = OUT / f"{model.name}.keras"
    cb = [keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True),
          keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]
    return model.fit(train_ds if tr is None else tr, validation_data=val_ds if va is None else va,
                     epochs=epochs, callbacks=cb).history

def merge(*hists):
    return {k: sum((h.get(k, []) for h in hists), []) for k in hists[0]}

def predict(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model.predict_on_batch(x)); y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

histories, results, trained = {}, [], {}

def evaluate(model, train_minutes):
    y_true, y_prob = predict(model, test_ds)
    y_pred = y_prob.argmax(1)
    x, _ = next(iter(test_ds))
    model.predict_on_batch(x)  # warm-up
    t = time.perf_counter(); model.predict_on_batch(x)
    ms = (time.perf_counter() - t) / len(x) * 1000
    path = OUT / f"{model.name}.keras"
    model.save(path)
    row = {"model": model.name,
           "test_accuracy": np.mean(y_pred == y_true),
           "macro_f1": f1_score(y_true, y_pred, average="macro"),
           "params_M": model.count_params() / 1e6,
           "size_MB": path.stat().st_size / 1e6,
           "ms_per_image": ms,
           "train_minutes": train_minutes}
    results.append(row); trained[model.name] = model
    print(row)
    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

def train_transfer(name, augment=True):
    model, base = build_transfer(name, augment)
    t = time.time()
    compile_model(model, 1e-3)
    h1 = fit(model, HEAD_EPOCHS)
    unfreeze_top(base)
    compile_model(model, 1e-5)
    h2 = fit(model, FINETUNE_EPOCHS)
    histories[model.name] = merge(h1, h2)
    evaluate(model, (time.time() - t) / 60)
""")

md("## 7. Baseline: custom CNN from scratch")
code("""
cnn = build_cnn()
cnn.summary()
t = time.time()
compile_model(cnn, 1e-3)
histories[cnn.name] = fit(cnn, CNN_EPOCHS)
evaluate(cnn, (time.time() - t) / 60)
""")

md("## 8. Transfer learning: fine-tune three pretrained backbones")
code("""
for name in BACKBONES:
    train_transfer(name)
""")

md("## 9. Ablation: MobileNetV2 without data augmentation")
code("""
if RUN_ABLATION:
    train_transfer("MobileNetV2", augment=False)
""")

md("## 10. Compare models")
code("""
df = pd.DataFrame(results).set_index("model")
display(df.round(4))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, title in zip(axes, ["test_accuracy", "size_MB", "ms_per_image"],
                          ["Test accuracy", "Model size (MB)", "Inference time (ms / image)"]):
    df[col].plot.bar(ax=ax, color="tab:green"); ax.set_title(title); ax.tick_params(axis="x", rotation=30)
axes[0].set_ylim(df["test_accuracy"].min() - 0.05, 1.0)
plt.tight_layout(); plt.savefig(OUT / "model_comparison.png", dpi=120); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for name, h in histories.items():
    axes[0].plot(h["val_accuracy"], label=name)
    axes[1].plot(h["val_loss"], label=name)
axes[0].set_title("Validation accuracy"); axes[1].set_title("Validation loss")
for ax in axes:
    ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=120); plt.show()
""")

md("### Confusion matrix and per-class report for the best model")
code("""
best_name = df["test_accuracy"].idxmax()
best = trained[best_name]
print("Best model:", best_name)

y_true, y_prob = predict(best, test_ds)
y_pred = y_prob.argmax(1)
report = classification_report(y_true, y_pred, labels=range(NUM_CLASSES), target_names=class_names, digits=4, zero_division=0)
print(report)
(OUT / "classification_report.txt").write_text(report)

cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES), normalize="true")
plt.figure(figsize=(16, 14))
plt.imshow(cm, cmap="Greens"); plt.colorbar()
short = [c.replace("___", ": ").replace("_", " ")[:28] for c in class_names]
plt.xticks(range(NUM_CLASSES), short, rotation=90, fontsize=7)
plt.yticks(range(NUM_CLASSES), short, fontsize=7)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix (normalized) - {best_name}")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=120); plt.show()
""")

md("""
## 11. Real-world test: PlantDoc field photos
PlantVillage photos are single leaves on a plain background. PlantDoc photos are taken in the field.
Only PlantDoc classes that also exist in PlantVillage are used.
""")
code("""
def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())

# PlantDoc folder -> keywords that identify the matching PlantVillage class
PLANTDOC_MAP = {
    "Apple Scab Leaf": ("apple", "scab"),
    "Apple leaf": ("apple", "healthy"),
    "Apple rust leaf": ("apple", "rust"),
    "Bell_pepper leaf": ("pepper", "healthy"),
    "Bell_pepper leaf spot": ("pepper", "bacterialspot"),
    "Blueberry leaf": ("blueberry", "healthy"),
    "Cherry leaf": ("cherry", "healthy"),
    "Corn Gray leaf spot": ("corn", "grayleafspot"),
    "Corn leaf blight": ("corn", "northernleafblight"),
    "Corn rust leaf": ("corn", "commonrust"),
    "Peach leaf": ("peach", "healthy"),
    "Potato leaf early blight": ("potato", "earlyblight"),
    "Potato leaf late blight": ("potato", "lateblight"),
    "Raspberry leaf": ("raspberry", "healthy"),
    "Soyabean leaf": ("soybean", "healthy"),
    "Squash Powdery mildew leaf": ("squash", "powderymildew"),
    "Strawberry leaf": ("strawberry", "healthy"),
    "Tomato Early blight leaf": ("tomato", "earlyblight"),
    "Tomato Septoria leaf spot": ("tomato", "septorialeafspot"),
    "Tomato leaf": ("tomato", "healthy"),
    "Tomato leaf bacterial spot": ("tomato", "bacterialspot"),
    "Tomato leaf late blight": ("tomato", "lateblight"),
    "Tomato leaf mosaic virus": ("tomato", "mosaicvirus"),
    "Tomato leaf yellow virus": ("tomato", "yellowleafcurl"),
    "Tomato mold leaf": ("tomato", "leafmold"),
    "Tomato two spotted spider mites leaf": ("tomato", "spidermite"),
    "grape leaf": ("grape", "healthy"),
    "grape leaf black rot": ("grape", "blackrot"),
}
MAP_BY_NORM = {norm(k): v for k, v in PLANTDOC_MAP.items()}

def pv_class(crop, disease):
    for i, c in enumerate(class_names):
        plant, _, cond = c.partition("___")
        if norm(plant).startswith(crop) and disease in norm(cond):
            return i
    return None

def is_valid_image(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False

def load_plantdoc(split):
    p, y, skipped = [], [], set()
    for d in sorted((PD_ROOT / split).iterdir()):
        if not d.is_dir():
            continue
        keys = MAP_BY_NORM.get(norm(d.name))
        idx = pv_class(*keys) if keys else None
        if idx is None:
            skipped.add(d.name); continue
        for f in list_images(d):
            if is_valid_image(f):
                p.append(str(f)); y.append(idx)
    if skipped:
        print(f"[{split}] folders without a PlantVillage match (skipped):", sorted(skipped))
    return np.array(p), np.array(y)

pd_tr_p, pd_tr_y = load_plantdoc("train")
pd_te_p, pd_te_y = load_plantdoc("test")
print(f"PlantDoc mapped images: train {len(pd_tr_p)}  test {len(pd_te_p)}  "
      f"classes {len(set(pd_te_y))}")
pd_test_ds = make_ds(pd_te_p, pd_te_y)
""")

code("""
def accuracy_on(model, ds):
    y_true, y_prob = predict(model, ds)
    return np.mean(y_prob.argmax(1) == y_true)

real_world = []
for name, model in trained.items():
    real_world.append({"model": name,
                       "PlantVillage_test": df.loc[name, "test_accuracy"],
                       "PlantDoc_field_photos": accuracy_on(model, pd_test_ds)})
rw = pd.DataFrame(real_world).set_index("model")
display(rw.round(4))
""")

md("""
## 12. Domain adaptation: fine-tune the best model on field photos
Train on PlantDoc field photos mixed with a sample of PlantVillage images,
so the model learns messy backgrounds without forgetting the lab images.
""")
code("""
mix_idx = np.random.default_rng(SEED).choice(len(tr_p), size=min(6000, len(tr_p)), replace=False)
mix_p = np.concatenate([pd_tr_p, tr_p[mix_idx]])
mix_y = np.concatenate([pd_tr_y, tr_y[mix_idx]])

adapted = keras.models.clone_model(best)
adapted.set_weights(best.get_weights())
adapted = keras.Model(adapted.inputs, adapted.outputs, name=best_name + "_plantdoc")
compile_model(adapted, 1e-5)
t = time.time()
# validation = PlantVillage val set, so early stopping protects lab accuracy
histories[adapted.name] = fit(adapted, PLANTDOC_EPOCHS, tr=make_ds(mix_p, mix_y, shuffle=True))

rw.loc[adapted.name] = {"PlantVillage_test": accuracy_on(adapted, test_ds),
                        "PlantDoc_field_photos": accuracy_on(adapted, pd_test_ds)}
display(rw.round(4))
rw.to_csv(OUT / "real_world_results.csv")

rw.plot.bar(figsize=(12, 4), rot=30, title="Lab images vs real field photos")
plt.ylabel("accuracy"); plt.tight_layout(); plt.savefig(OUT / "real_world_comparison.png", dpi=120); plt.show()
""")

md("""
## 13. Save the final model for the app
The adapted model is used in the app because it handles real photos better.
""")
code("""
final = adapted
final.save(OUT / "best_model.keras")
with open(OUT / "model_info.json", "w") as f:
    json.dump({"model": final.name, "img_size": IMG_SIZE,
               "tensorflow": tf.__version__, "keras": keras.__version__,
               "confidence_threshold": CONFIDENCE_THRESHOLD}, f, indent=1)
print("Saved to", OUT / "best_model.keras")
print("Install the same TensorFlow version locally:  pip install tensorflow==" + tf.__version__)
""")

md("## 14. Grad-CAM: where does the model look?")
code("""
def grad_cam(model, img):
    \"\"\"img: HxWx3 array (0-255). Returns (heatmap 0-1, class index, probabilities).\"\"\"
    model_layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    gap = next(i for i, l in enumerate(model_layers) if isinstance(l, keras.layers.GlobalAveragePooling2D))
    x = tf.convert_to_tensor(img[None], tf.float32)
    with tf.GradientTape() as tape:
        for i, layer in enumerate(model_layers):
            x = layer(x, training=False)
            if i == gap - 1:  # last convolutional feature map
                conv = x
                tape.watch(conv)
        probs = tf.cast(x, tf.float32)
        idx = int(tf.argmax(probs[0]))
        score = probs[:, idx]
    grads = tf.cast(tape.gradient(score, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, idx, probs[0].numpy()

def overlay(img, cam, alpha=0.45):
    cam = np.array(Image.fromarray(np.uint8(cam * 255)).resize(img.shape[1::-1], Image.BILINEAR)) / 255.0
    heat = plt.cm.jet(cam)[..., :3] * 255
    return np.uint8((1 - alpha) * img + alpha * heat)

def show_gradcam(model, items, title, fname):
    plt.figure(figsize=(14, 6))
    for k, (p, y) in enumerate(items):
        img, _ = load_image(p, y)
        img = img.numpy()
        cam, idx, probs = grad_cam(model, img)
        plt.subplot(2, len(items), k + 1)
        plt.imshow(img.astype("uint8")); plt.axis("off")
        plt.title("True: " + class_names[y].split("___")[-1][:22], fontsize=8)
        plt.subplot(2, len(items), len(items) + k + 1)
        plt.imshow(overlay(img, cam)); plt.axis("off")
        plt.title(f"Pred: {class_names[idx].split('___')[-1][:18]} ({probs[idx]:.0%})", fontsize=8)
    plt.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=120); plt.show()

rng = np.random.default_rng(1)
show_gradcam(final, [(te_p[i], te_y[i]) for i in rng.choice(len(te_p), 6, replace=False)],
             "Grad-CAM on PlantVillage test images", "gradcam_plantvillage.png")
show_gradcam(final, [(pd_te_p[i], pd_te_y[i]) for i in rng.choice(len(pd_te_p), 6, replace=False)],
             "Grad-CAM on PlantDoc field photos", "gradcam_plantdoc.png")
""")

md("""
## Done
Everything is in `MyDrive/plant_disease/`:
- `best_model.keras`, `class_names.json`, `model_info.json` → copy to the project's `models/` folder
- `results.csv`, `real_world_results.csv`, `classification_report.txt` → report tables
- `*.png` → report figures
""")

nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
out = pathlib.Path(__file__).resolve().parent.parent / "train.ipynb"
nbf.write(nb, out)
print("wrote", out)
