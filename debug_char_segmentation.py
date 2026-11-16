#!/usr/bin/env python3
"""
debug_char_segmentation.py - Diagnostics & Character Segmentation Visualizer

Segments individual characters from a vehicle license plate:
  1. Cropped plate extraction
  2. Plate cleaning via adaptive Gaussian thresholding
  3. HSV V-channel extraction & adaptive binarization
  4. Connected-component labeling (skimage.measure.label) & convex hull filtering
  5. Left-to-right character patch extraction & disk export
"""

import argparse
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from src.plate_finder import PlateFinder
from debug_plate_finder import create_synthetic_test_image


def debug_segmentation(
    image_path: str,
    output_dir: str = "output/segmentation",
    fixed_width: int = 400,
    min_chars: int = 4,
    max_chars: int = 12,
):
    if cv2 is None or np is None:
        print("Error: OpenCV and NumPy are required.", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve input image
    if not os.path.exists(image_path):
        if image_path == "data/sample_car.jpg" or not os.path.dirname(image_path):
            print(f"Notice: '{image_path}' not found. Generating synthetic car image...")
            create_synthetic_test_image(image_path)
        else:
            print(f"Error: Input image not found: '{image_path}'", file=sys.stderr)
            sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not decode image at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    chars_dir = os.path.join(output_dir, "chars")
    os.makedirs(chars_dir, exist_ok=True)

    finder = PlateFinder()

    # 2. Extract plate region
    # If the input image is already a cropped plate (e.g. aspect ratio > 1.8 and small height), use directly
    img_h, img_w = img.shape[:2]
    aspect = (img_w / float(img_h)) if img_h > 0 else 0

    if aspect >= 2.0 and img_h < 300:
        plate_crop = img.copy()
        print(f"Input image appears to be an already cropped plate ({img_w}x{img_h}).")
    else:
        # Locate plate in full car image
        candidates = finder.find_possible_plates(img)
        if not candidates:
            print("No plate candidate detected in input image; attempting segmentation on full image as fallback.")
            plate_crop = img.copy()
        else:
            # Pick best candidate (first or largest)
            best_cand = candidates[0]
            rect = best_cand["rect"]
            box = cv2.boxPoints(rect)
            x_min = max(0, int(np.min(box[:, 0])))
            x_max = min(img_w, int(np.max(box[:, 0])))
            y_min = max(0, int(np.min(box[:, 1])))
            y_max = min(img_h, int(np.max(box[:, 1])))
            plate_crop = img[y_min:y_max, x_min:x_max]
            print(f"Extracted plate candidate region: ({x_min}, {y_min}) to ({x_max}, {y_max}).")

    p_crop = os.path.join(output_dir, "01_cropped_plate.jpg")
    cv2.imwrite(p_crop, plate_crop)
    print(f"  [1/6] Saved Cropped Plate           -> {p_crop}")

    # 3. Clean plate
    cleaned_plate, clean_box = finder.clean_plate(plate_crop)
    p_clean = os.path.join(output_dir, "02_clean_plate.jpg")
    cv2.imwrite(p_clean, cleaned_plate)
    print(f"  [2/6] Saved Clean Plate             -> {p_clean} (Box: {clean_box})")

    # 4. Segment characters
    seg_res = finder.segment_chars(
        cleaned_plate,
        fixed_width=fixed_width,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    if seg_res["v_channel"] is not None:
        p_v = os.path.join(output_dir, "03_v_channel.jpg")
        cv2.imwrite(p_v, seg_res["v_channel"])
        print(f"  [3/6] Saved HSV V-Channel           -> {p_v}")

    if seg_res["thresh"] is not None:
        p_th = os.path.join(output_dir, "04_adaptive_thresh.jpg")
        cv2.imwrite(p_th, seg_res["thresh"])
        print(f"  [4/6] Saved Adaptive Binary Thresh  -> {p_th}")

    if seg_res["char_mask"] is not None:
        p_mask = os.path.join(output_dir, "05_char_candidates_mask.jpg")
        cv2.imwrite(p_mask, seg_res["char_mask"])
        print(f"  [5/6] Saved Convex Hull Mask        -> {p_mask}")

    # 5. Export individual character images & annotate plate
    characters = seg_res["characters"]
    count = seg_res["count"]
    annotated_plate = seg_res["plate_resized"].copy()

    print(f"\n--- Segmented {count} Character Candidates (Valid Range: {min_chars}-{max_chars}) ---")

    for i, ch in enumerate(characters):
        x, y, w, h = ch["bbox"]
        patch = ch["patch"]
        binary_patch = ch["binary"]

        # Save individual character patch
        char_filename = os.path.join(chars_dir, f"char_{i+1:02d}.jpg")
        char_bin_filename = os.path.join(chars_dir, f"char_{i+1:02d}_bin.jpg")
        cv2.imwrite(char_filename, patch)
        cv2.imwrite(char_bin_filename, binary_patch)

        # Draw bounding box and index on annotated plate
        cv2.rectangle(annotated_plate, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            annotated_plate,
            str(i + 1),
            (x + 2, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        aspect_r = (w / float(h)) if h > 0 else 0
        print(f"  Character #{i+1:02d}: BBox=({x}, {y}, {w}, {h}) | AspectRatio={aspect_r:.2f} -> {char_filename}")

    p_annotated = os.path.join(output_dir, "06_annotated_plate.jpg")
    cv2.imwrite(p_annotated, annotated_plate)
    print(f"\n  [6/6] Saved Annotated Plate Image    -> {p_annotated}")

    status_str = "SUCCESS (within expected range)" if seg_res["is_valid_count"] else "WARNING (outside expected range)"
    print(f"\nCharacter Validation Status: {status_str}")
    print(f"All character patches and debug visualizations written to '{output_dir}/'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and segment individual characters from a vehicle license plate."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/sample_car.jpg",
        help="Path to input car image or cropped plate (default: data/sample_car.jpg).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/segmentation",
        help="Output directory for character patches (default: output/segmentation).",
    )
    parser.add_argument("--fixed-width", type=int, default=400, help="Normalized plate width in px.")
    parser.add_argument("--min-chars", type=int, default=4, help="Minimum expected character count.")
    parser.add_argument("--max-chars", type=int, default=12, help="Maximum expected character count.")

    args = parser.parse_args()

    debug_segmentation(
        image_path=args.image,
        output_dir=args.output,
        fixed_width=args.fixed_width,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
