#!/usr/bin/env python3
"""
Phase 2: Frame Preprocessing & Vertical Edge Extraction

Visualizes the transformation pipeline:
1. Original BGR Frame
2. Grayscale Conversion
3. Gaussian Blur Smoothing
4. Vertical Edge Extraction (Sobel-X)
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

from src.preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Preprocess video frames and extract vertical edges."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the video file to be processed."
    )
    parser.add_argument(
        "--view",
        type=str,
        default="grid",
        choices=["original", "gray", "blurred", "edges", "grid"],
        help="Initial view mode: 'original', 'gray', 'blurred', 'edges', or 'grid' (default: grid)."
    )
    return parser.parse_args()


def create_annotated_tile(image: "np.ndarray", title: str) -> "np.ndarray":
    """Ensures 3 channels and draws a clear title header on top."""
    # Convert single-channel to 3-channel for stacking
    if len(image.shape) == 2:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        tile = image.copy()

    # Draw semi-transparent header bar
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, tile, 0.3, 0, tile)

    # Put title text
    cv2.putText(
        tile,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def build_grid_view(proc_dict: dict, display_width: int, display_height: int) -> "np.ndarray":
    """Combines all 4 preprocessing stages into an annotated 2x2 comparison grid."""
    t_orig = create_annotated_tile(proc_dict["original"], "1: Original (BGR)")
    t_gray = create_annotated_tile(proc_dict["gray"], "2: Grayscale (Y-channel)")
    t_blur = create_annotated_tile(proc_dict["blurred"], "3: Gaussian Blur (5x5)")
    t_edge = create_annotated_tile(proc_dict["vertical_edges"], "4: Vertical Edges (Sobel-X)")

    top_row = np.hstack([t_orig, t_gray])
    bottom_row = np.hstack([t_blur, t_edge])
    grid = np.vstack([top_row, bottom_row])

    # Scale down to original single-frame dimensions so it fits comfortably on screen
    return cv2.resize(grid, (display_width, display_height), interpolation=cv2.INTER_AREA)


def run_pipeline(video_path: str, initial_view: str = "grid"):
    # 1. Cleanly check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Error: Specified path is not a valid file: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Check dependencies
    if cv2 is None or np is None:
        print("Error: Required dependencies (OpenCV/NumPy) are not installed.", file=sys.stderr)
        print("Please install requirements using: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    # 3. Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: OpenCV could not open video file: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target display size (cap to max 1280x720 for comfortable desktop viewing)
    scale = min(1.0, 1280 / max(orig_w, 1), 720 / max(orig_h, 1))
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    preprocessor = Preprocessor(blur_ksize=(5, 5), sobel_ksize=3)
    current_view = initial_view
    paused = False

    print("=== Phase 2: Frame Preprocessing Controls ===")
    print(" [1] Original BGR View")
    print(" [2] Grayscale View")
    print(" [3] Gaussian Blurred View")
    print(" [4] Vertical Edges (Sobel-X) View")
    print(" [g] 2x2 Comparison Grid View")
    print(" [SPACE] Pause / Resume")
    print(" [q] Quit")
    print("=============================================")

    window_name = "Number Plate Detector - Phase 2"
    frame_count = 0

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print(f"\nEnd of video stream. Processed {frame_count} frames.")
                    break
                frame_count += 1
                proc = preprocessor.process(frame)

            # Determine what image to render based on current view
            if current_view == "grid":
                display_img = build_grid_view(proc, disp_w, disp_h)
            elif current_view == "gray":
                display_img = create_annotated_tile(proc["gray"], "2: Grayscale")
            elif current_view == "blurred":
                display_img = create_annotated_tile(proc["blurred"], "3: Gaussian Blur")
            elif current_view == "edges":
                display_img = create_annotated_tile(proc["vertical_edges"], "4: Vertical Edges (Sobel-X)")
            else:
                display_img = create_annotated_tile(proc["original"], "1: Original (BGR)")

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(delay if not paused else 50) & 0xFF

            if key == ord('q'):
                print(f"\nPlayback stopped by user at frame {frame_count}.")
                break
            elif key == ord(' '):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"[{status}] Frame {frame_count}")
            elif key == ord('1') or key == ord('o'):
                current_view = "original"
            elif key == ord('2'):
                current_view = "gray"
            elif key == ord('3'):
                current_view = "blurred"
            elif key == ord('4') or key == ord('e'):
                current_view = "edges"
            elif key == ord('g'):
                current_view = "grid"

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_pipeline(args.video, initial_view=args.view)


if __name__ == "__main__":
    main()
