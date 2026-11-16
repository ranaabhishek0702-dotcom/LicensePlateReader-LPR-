#!/usr/bin/env python3
"""
train.py - Alphanumeric Character Classifier Training

Trains a lightweight Convolutional Neural Network (CNN) to classify
isolated license plate characters across 36 classes (0-9, A-Z).

Supported Datasets:
  1. Synthetic Font Generator (Default, zero download):
     Generates augmented character glyphs using system fonts with rotations,
     blur, scaling, and noise.
  2. EMNIST Balanced / ByClass (Large download ~535MB):
     Downloads the NIST/EMNIST benchmark dataset. Requires explicit --confirm-download flag.
"""

import argparse
import os
import sys
import json
from typing import Tuple

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.ocr import ALPHANUMERIC_CHARS, LABEL_TO_CHAR, CHAR_TO_LABEL


def generate_synthetic_char_dataset(
    samples_per_class: int = 400,
    img_size: Tuple[int, int] = (28, 28),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes an augmented character dataset locally without external downloads.
    Generates variations in scale, rotation, stroke thickness, and noise.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for dataset generation.")

    print(f"Generating synthetic alphanumeric dataset ({len(ALPHANUMERIC_CHARS)} classes, {samples_per_class} samples/class)...")

    X = []
    y = []

    # Fonts available in OpenCV for rendering
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    for label_idx, char in enumerate(ALPHANUMERIC_CHARS):
        for _ in range(samples_per_class):
            # 1. Base canvas (64x64 for rendering headroom, scaled down to 28x28)
            canvas = np.zeros((64, 64), dtype=np.uint8)

            font = fonts[np.random.randint(0, len(fonts))]
            font_scale = np.random.uniform(1.1, 1.5)
            thickness = np.random.randint(2, 4)

            # Get text bounding size to center it
            (text_w, text_h), baseline = cv2.getTextSize(char, font, font_scale, thickness)
            tx = max(2, (64 - text_w) // 2 + np.random.randint(-3, 4))
            ty = max(text_h + 2, (64 + text_h) // 2 + np.random.randint(-3, 4))

            # Draw white character on black background
            cv2.putText(canvas, char, (tx, ty), font, font_scale, 255, thickness, cv2.LINE_AA)

            # 2. Random Affine transformations (rotation +/- 8 degrees)
            angle = np.random.uniform(-8.0, 8.0)
            rot_mat = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            canvas = cv2.warpAffine(canvas, rot_mat, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 3. Random morphological erosion or dilation to vary stroke width
            if np.random.rand() > 0.5:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                if np.random.rand() > 0.5:
                    canvas = cv2.dilate(canvas, kernel, iterations=1)
                else:
                    canvas = cv2.erode(canvas, kernel, iterations=1)

            # 4. Subtle Gaussian blur and sensor noise
            if np.random.rand() > 0.5:
                canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

            noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 5. Crop tight bounding box and resize to canonical 28x28
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw > 2 and bh > 2:
                    patch = canvas[by:by+bh, bx:bx+bw]

                    # Square pad
                    max_dim = max(bw, bh)
                    pt = (max_dim - bh) // 2
                    pb = max_dim - bh - pt
                    pl = (max_dim - bw) // 2
                    pr = max_dim - bw - pl
                    padded = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
                    final_glyph = cv2.resize(padded, img_size, interpolation=cv2.INTER_AREA)
                else:
                    final_glyph = cv2.resize(canvas, img_size, interpolation=cv2.INTER_AREA)
            else:
                final_glyph = cv2.resize(canvas, img_size, interpolation=cv2.INTER_AREA)

            X.append(final_glyph)
            y.append(label_idx)

    X = np.array(X, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(y, dtype="int32")

    # Shuffle dataset
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def build_cnn_model(input_shape=(28, 28, 1), num_classes=36):
    """
    Lightweight 2-block ConvNet tailored for isolated alphanumeric glyphs.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(
    dataset_type: str = "synthetic",
    confirm_download: bool = False,
    output_model_path: str = "models/char_cnn.keras",
    epochs: int = 15,
    batch_size: int = 64,
):
    if tf is None:
        print("Error: TensorFlow is required for training. Install via: pip install tensorflow", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    if dataset_type == "emnist":
        if not confirm_download:
            print(
                "\n[Safeguard Prompt] EMNIST is a public dataset (~535 MB archive).\n"
                "To download and use EMNIST, re-run with the flag: --confirm-download\n"
                "Alternatively, use the default local synthetic generator (0 MB download): python3 train.py\n",
                file=sys.stderr
            )
            sys.exit(0)

        print("Downloading and preparing EMNIST dataset...")
        try:
            import tensorflow_datasets as tfds
            ds_train, ds_val = tfds.load(
                'emnist/balanced',
                split=['train', 'test'],
                as_supervised=True,
                batch_size=-1
            )
            # Convert TFDS to numpy arrays
            X_train_raw, y_train_raw = tfds.as_numpy(ds_train)
            X_val_raw, y_val_raw = tfds.as_numpy(ds_val)

            # EMNIST Balanced has 47 classes (0-9 are 0-9, 10-35 are A-Z)
            # Filter only the 36 classes we care about (0 to 35)
            train_mask = y_train_raw < len(ALPHANUMERIC_CHARS)
            val_mask = y_val_raw < len(ALPHANUMERIC_CHARS)

            X_train = (X_train_raw[train_mask].astype("float32") / 255.0).reshape(-1, 28, 28, 1)
            y_train = y_train_raw[train_mask]
            X_val = (X_val_raw[val_mask].astype("float32") / 255.0).reshape(-1, 28, 28, 1)
            y_val = y_val_raw[val_mask]
            print(f"EMNIST filtered: {len(X_train)} train, {len(X_val)} validation samples.")
        except Exception as e:
            print(f"Failed to load EMNIST via tensorflow_datasets: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Default: Synthetic high-performance plate font generator
        X, y = generate_synthetic_char_dataset(samples_per_class=350)
        split = int(0.85 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        print(f"Dataset ready: {len(X_train)} training samples, {len(X_val)} validation samples.")

    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=len(ALPHANUMERIC_CHARS))
    model.summary()

    print(f"\nTraining CNN across {epochs} epochs...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )

    model.save(output_model_path)
    print(f"\nTrained model successfully saved to: {output_model_path}")

    # Save class mapping
    label_file = os.path.splitext(output_model_path)[0] + "_labels.json"
    with open(label_file, "w") as f:
        json.dump(LABEL_TO_CHAR, f, indent=2)
    print(f"Class labels saved to: {label_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train isolated character classification CNN for ANPR."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "emnist"],
        default="synthetic",
        help="Dataset choice: 'synthetic' (zero download) or 'emnist' (~535MB).",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Explicit confirmation flag required before downloading large datasets like EMNIST.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (default: 12).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/char_cnn.keras",
        help="Output path for serialized Keras model (default: models/char_cnn.keras).",
    )

    args = parser.parse_args()

    train_model(
        dataset_type=args.dataset,
        confirm_download=args.confirm_download,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
