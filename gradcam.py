"""Grad-CAM for the models built in train.ipynb (same code as the notebook)."""
import numpy as np
import tensorflow as tf
import keras
from matplotlib import colormaps
from PIL import Image


def grad_cam(model, img, class_idx=None):
    """img: HxWx3 array (0-255). Explains class_idx (default: top class).
    Returns (heatmap 0-1, class index, probabilities)."""
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
        idx = int(tf.argmax(probs[0])) if class_idx is None else int(class_idx)
        score = probs[:, idx]
    grads = tf.cast(tape.gradient(score, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, idx, probs[0].numpy()


def overlay(img, cam, alpha=0.45):
    """Blend a heatmap over an HxWx3 uint8 image."""
    cam = np.array(Image.fromarray(np.uint8(cam * 255)).resize(img.shape[1::-1], Image.BILINEAR)) / 255.0
    heat = colormaps["jet"](cam)[..., :3] * 255
    return np.uint8((1 - alpha) * img + alpha * heat)
