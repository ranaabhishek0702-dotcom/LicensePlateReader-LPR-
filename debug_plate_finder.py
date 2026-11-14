#!/usr/bin/env python3
"""
debug_plate_finder.py - Debug script for Stage 1 Plate Detection

Runs PlateFinder on a still image and writes each stage to the 'output/' folder:
  1. 01_blur.jpg               - Gaussian Blur (7x7)
  2. 02_gray.jpg               - Grayscale
  3. 03_sobel.jpg              - Vertical Edges (Sobel-X)
  4. 04_threshold.jpg          - Otsu Threshold
  5. 05_closed.jpg             - Morphological Close (22x3)
  6. 06_all_contours.jpg       - All candidate contours (yellow/red)
  7. 07_validated_contours.jpg - Validated plates drawn in green
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


def create_synthetic_test_image(save_path: str):
    """
    Generates a realistic synthetic test frame with a car bumper and
    license plate so the script can be tested without an external video file.
    """
    if cv2 is None or np is None:
        return

    # 1. Dark car bumper background (800x500)
    img = np.full((500, 800, 3), (35, 35, 40), dtype=np.uint8)

    # Add subtle car body curves / horizontal bumper lines
    cv2.line(img, (50, 150), (750, 150), (60, 60, 65), 3)
    cv2.line(img, (20, 350), (780, 350), (50, 50, 55), 4)

    # Red taillights on sides
    cv2.rectangle(img, (60, 180), (160, 240), (20, 20, 180), -1)
    cv2.rectangle(img, (640, 180), (740, 240), (20, 20, 180), -1)

    # 2. Car License Plate (White rectangle with aspect ratio ~4.0)
    # Center: (400, 280), Dimensions: 240 x 60 (Aspect ratio = 4.0)
    plate_w, plate_h = 240, 60
    px1 = 400 - plate_w // 2
    py1 = 280 - plate_h // 2
    px2 = px1 + plate_w
    py2 = py1 + plate_h

    # White plate surface
    cv2.rectangle(img, (px1, py1), (px2, py2), (240, 240, 240), -1)
    # Thin dark plate border
    cv2.rectangle(img, (px1, py1), (px2, py2), (20, 20, 20), 2)

    # Bold vertical characters on plate: "DL 01 AB 1234"
    text = "DL 01 AB 1234"
    cv2.putText(
        img,
        text,
        (px1 + 18, py1 + 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (15, 15, 15),
        3,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"Generated synthetic test image at: {save_path}")


def debug_plate_detection(
    image_path: str,
    output_dir: str = "output",
    min_area: float = 800.0,
    max_area: float = 50000.0,
    min_ratio: float = 2.0,
    max_ratio: float = 6.0,
    max_tilt: float = 15.0,
):
    if cv2 is None or np is None:
        print("Error: OpenCV or NumPy is not installed.", file=sys.stderr)
        sys.exit(1)

    # Check if input image exists; if not and path is sample, generate it
    if not os.path.exists(image_path):
        if image_path == "data/sample_car.jpg" or not os.path.dirname(image_path):
            print(f"Notice: '{image_path}' not found. Generating a synthetic test car image...")
            create_synthetic_test_image(image_path)
        else:
            print(f"Error: Input image not found: '{image_path}'", file=sys.stderr)
            sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not decode image at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing '{image_path}' ({img.shape[1]}x{img.shape[0]} px)...")

    # Initialize PlateFinder
    finder = PlateFinder(
        min_area=min_area,
        max_area=max_area,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        max_tilt=max_tilt,
    )

    # 1. Run preprocessing steps
    steps = finder.preprocess_steps(img)

    # Save intermediate images
    p_blur = os.path.join(output_dir, "01_blur.jpg")
    p_gray = os.path.join(output_dir, "02_gray.jpg")
    p_sobel = os.path.join(output_dir, "03_sobel.jpg")
    p_thresh = os.path.join(output_dir, "04_threshold.jpg")
    p_closed = os.path.join(output_dir, "05_closed.jpg")

    cv2.imwrite(p_blur, steps["blur"])
    cv2.imwrite(p_gray, steps["gray"])
    cv2.imwrite(p_sobel, steps["sobel"])
    cv2.imwrite(p_thresh, steps["threshold"])
    cv2.imwrite(p_closed, steps["closed"])

    print(f"  [1/7] Saved Gaussian Blur (7x7)   -> {p_blur}")
    print(f"  [2/7] Saved Grayscale Image       -> {p_gray}")
    print(f"  [3/7] Saved Sobel-X Edges         -> {p_sobel}")
    print(f"  [4/7] Saved Otsu Threshold Binary -> {p_thresh}")
    print(f"  [5/7] Saved Morphological Closed  -> {p_closed}")

    # 2. Extract external contours
    contours = finder.extract_contours(steps["closed"])
    print(f"\nExtracted {len(contours)} external contours.")

    # 3. Draw all contours (yellow outline)
    all_cnt_img = img.copy()
    cv2.drawContours(all_cnt_img, contours, -1, (0, 230, 255), 2)
    p_all_cnt = os.path.join(output_dir, "06_all_contours.jpg")
    cv2.imwrite(p_all_cnt, all_cnt_img)
    print(f"  [6/7] Saved All Contours          -> {p_all_cnt}")

    # 4. Validate contours
    validated_img = img.copy()
    validated_count = 0

    print("\n--- Contour Validation Details ---")
    for i, cnt in enumerate(contours):
        if len(cnt) < 4:
            continue
        rect = cv2.minAreaRect(cnt)
        is_valid = finder.validate_ratio(rect)

        # Calculate metrics for logging
        box = cv2.boxPoints(rect)
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        l1, l2 = float(np.hypot(edge1[0], edge1[1])), float(np.hypot(edge2[0], edge2[1]))
        w_box, h_box = max(l1, l2), min(l1, l2)
        area = w_box * h_box
        ratio = (w_box / h_box) if h_box > 0 else 0
        long_edge = edge1 if l1 >= l2 else edge2
        tilt = float(np.degrees(np.arctan2(abs(long_edge[1]), abs(long_edge[0]))))

        status_str = "VALID [PLATE]" if is_valid else "REJECTED"
        print(
            f"  Contour #{i+1:02d}: Area={area:7.1f} | Ratio={ratio:4.2f} | Tilt={tilt:4.1f}° -> {status_str}"
        )

        if is_valid:
            validated_count += 1
            pts = np.intp(box)
            # Draw green bounding box around validated candidate
            cv2.drawContours(validated_img, [pts], 0, (0, 255, 0), 3)

            # Draw tag text
            tag = f"Plate candidate: {w_box:.0f}x{h_box:.0f} (r={ratio:.1f})"
            label_y = max(20, int(rect[0][1] - h_box / 2 - 8))
            label_x = max(10, int(rect[0][0] - w_box / 2))
            cv2.putText(
                validated_img,
                tag,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    p_validated = os.path.join(output_dir, "07_validated_contours.jpg")
    cv2.imwrite(p_validated, validated_img)
    print(f"\n  [7/7] Saved Validated Contours   -> {p_validated}")
    print(f"\nResult: Found {validated_count} valid license plate candidate(s) out of {len(contours)} contours.")
    print(f"All debug images successfully written to '{output_dir}/'.")


def main():
    parser = argparse.ArgumentParser(
        description="Debug script: run PlateFinder pipeline on a still image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/sample_car.jpg",
        help="Path to input still image (default: data/sample_car.jpg).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Directory to save debug images (default: output).",
    )
    parser.add_argument("--min-area", type=float, default=800.0, help="Min plate area in pixels.")
    parser.add_argument("--max-area", type=float, default=50000.0, help="Max plate area in pixels.")
    parser.add_argument("--min-ratio", type=float, default=2.0, help="Min aspect ratio (w/h).")
    parser.add_argument("--max-ratio", type=float, default=6.0, help="Max aspect ratio (w/h).")
    parser.add_argument("--max-tilt", type=float, default=15.0, help="Max tilt angle in degrees.")

    args = parser.parse_args()

    debug_plate_detection(
        image_path=args.image,
        output_dir=args.output,
        min_area=args.min_area,
        max_area=args.max_area,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        max_tilt=args.max_tilt,
    )


if __name__ == "__main__":
    main()
