#!/usr/bin/env python3
"""
compare_ocr.py - Side-by-Side OCR Backend Comparison

Runs both recognition backends on segmented license plate characters:
  - Backend A: Tesseract OCR (single-character mode --psm 10)
  - Backend B: Custom Keras CNN Classifier
"""

import argparse
import os
import sys
import glob
from typing import List

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from src.plate_finder import PlateFinder
from src.ocr import TesseractRecognizer, KerasCNNRecognizer
from debug_plate_finder import create_synthetic_test_image


def load_or_segment_characters(image_path: str, chars_dir: str) -> List[np.ndarray]:
    """Loads existing character patches from disk or executes segmentation."""
    existing_patches = sorted(glob.glob(os.path.join(chars_dir, "char_[0-9][0-9].jpg")))
    if existing_patches:
        print(f"Loading {len(existing_patches)} pre-segmented character patches from '{chars_dir}'...")
        images = [cv2.imread(p) for p in existing_patches if cv2.imread(p) is not None]
        if images:
            return images

    print(f"No existing character patches in '{chars_dir}'. Running segmentation on '{image_path}'...")
    if not os.path.exists(image_path):
        create_synthetic_test_image(image_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'", file=sys.stderr)
        sys.exit(1)

    finder = PlateFinder()
    candidates = finder.find_possible_plates(img)

    if candidates:
        box = cv2.boxPoints(candidates[0]["rect"])
        h, w = img.shape[:2]
        x1 = max(0, int(np.min(box[:, 0])))
        x2 = min(w, int(np.max(box[:, 0])))
        y1 = max(0, int(np.min(box[:, 1])))
        y2 = min(h, int(np.max(box[:, 1])))
        plate_crop = img[y1:y2, x1:x2]
    else:
        plate_crop = img

    cleaned_plate, _ = finder.clean_plate(plate_crop)
    res = finder.segment_chars(cleaned_plate)
    return [ch["patch"] for ch in res["characters"]]


def compare_backends(
    image_path: str,
    chars_dir: str = "output/segmentation/chars",
    model_path: str = "models/char_cnn.keras",
):
    if cv2 is None or np is None:
        print("Error: OpenCV and NumPy are required.", file=sys.stderr)
        sys.exit(1)

    char_images = load_or_segment_characters(image_path, chars_dir)

    if not char_images:
        print("Error: No character images available for recognition.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEvaluating {len(char_images)} character patches across both OCR backends...\n")

    # 1. Initialize Backend A (Tesseract)
    backend_a_ready = True
    try:
        rec_a = TesseractRecognizer()
    except Exception as e:
        rec_a = None
        backend_a_ready = False

    # 2. Initialize Backend B (Keras CNN)
    backend_b_ready = True
    try:
        rec_b = KerasCNNRecognizer(model_path=model_path)
        if rec_b.model is None:
            backend_b_ready = False
    except Exception as e:
        rec_b = None
        backend_b_ready = False

    # 3. Process characters
    a_results = []
    b_results = []

    for idx, patch in enumerate(char_images):
        # Backend A
        if backend_a_ready and rec_a:
            try:
                char_a, conf_a = rec_a.recognize_char(patch)
            except Exception as e:
                char_a, conf_a = "N/A", 0.0
        else:
            char_a, conf_a = "N/A", 0.0
        a_results.append((char_a, conf_a))

        # Backend B
        if backend_b_ready and rec_b and rec_b.model is not None:
            try:
                char_b, conf_b = rec_b.recognize_char(patch)
            except Exception as e:
                char_b, conf_b = "N/A", 0.0
        else:
            char_b, conf_b = "N/A", 0.0
        b_results.append((char_b, conf_b))

    # 4. Render comparison table
    col_idx = "Index"
    col_a = "Backend A (Tesseract)"
    col_b = "Backend B (Keras CNN)"
    col_match = "Agreement"

    header = f"| {col_idx:^7} | {col_a:^24} | {col_b:^24} | {col_match:^11} |"
    divider = f"|{'-'*9}|{'-'*26}|{'-'*26}|{'-'*13}|"

    print("=" * len(header))
    print(header)
    print(divider)

    for i in range(len(char_images)):
        c_a, conf_a = a_results[i]
        c_b, conf_b = b_results[i]

        val_a = f"{c_a} (conf: {conf_a:.2f})" if c_a != "N/A" else "Not installed"
        val_b = f"{c_b} (conf: {conf_b:.2f})" if c_b != "N/A" else "Untrained model"

        if c_a != "N/A" and c_b != "N/A":
            match_str = "MATCH" if c_a == c_b else "MISMATCH"
        else:
            match_str = "PENDING"

        print(f"|  #{i+1:02d}   | {val_a:<24} | {val_b:<24} | {match_str:^11} |")

    print("=" * len(header))

    str_a = "".join([c[0] for c in a_results if c[0] != "N/A"])
    str_b = "".join([c[0] for c in b_results if c[0] != "N/A"])

    print("\n--- Final Recognized License Plate String ---")
    print(f"Backend A (Tesseract): {str_a if str_a else '[Requires pytesseract & tesseract binary]'}")
    print(f"Backend B (Keras CNN): {str_b if str_b else f'[Requires trained model: python3 train.py]'}")

    if not backend_a_ready or not backend_b_ready:
        print("\nSetup Notes:")
        if not backend_a_ready:
            print(" - To enable Backend A: pip install pytesseract (and install tesseract via: brew install tesseract)")
        if not backend_b_ready:
            print(f" - To enable Backend B: Run training with: python3 train.py (generates {model_path})")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of Tesseract vs Keras CNN character recognition."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/sample_car.jpg",
        help="Input vehicle image or cropped plate (default: data/sample_car.jpg).",
    )
    parser.add_argument(
        "--chars-dir",
        default="output/segmentation/chars",
        help="Directory with pre-segmented characters (default: output/segmentation/chars).",
    )
    parser.add_argument(
        "--model",
        default="models/char_cnn.keras",
        help="Path to trained Keras CNN model (default: models/char_cnn.keras).",
    )

    args = parser.parse_args()

    compare_backends(
        image_path=args.image,
        chars_dir=args.chars_dir,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
