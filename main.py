#!/usr/bin/env python3
"""
main.py - Real-Time Automatic Number Plate Recognition (ANPR)

Processes video streams in real-time, executing candidate plate localization
and providing multi-stage pipeline visualization.
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
from src.plate_finder import PlateFinder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time video pipeline for vehicle license plate detection."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the video file to be processed."
    )
    parser.add_argument(
        "--view",
        type=str,
        default="detected",
        choices=["detected", "grid", "original", "gray", "blurred", "edges"],
        help="Initial view mode: 'detected', 'grid', 'original', 'gray', 'blurred', or 'edges' (default: detected)."
    )
    parser.add_argument("--min-area", type=float, default=800.0, help="Min plate area in pixels.")
    parser.add_argument("--max-area", type=float, default=50000.0, help="Max plate area in pixels.")
    parser.add_argument("--min-ratio", type=float, default=2.0, help="Min aspect ratio (w/h).")
    parser.add_argument("--max-ratio", type=float, default=6.0, help="Max aspect ratio (w/h).")
    parser.add_argument("--max-tilt", type=float, default=15.0, help="Max tilt angle in degrees.")
    return parser.parse_args()


def create_annotated_tile(image: "np.ndarray", title: str) -> "np.ndarray":
    """Ensures 3 channels and draws an informative header overlay."""
    if len(image.shape) == 2:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        tile = image.copy()

    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, tile, 0.3, 0, tile)

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
    t_orig = create_annotated_tile(proc_dict["original"], "Original (BGR)")
    t_gray = create_annotated_tile(proc_dict["gray"], "Grayscale (Y-channel)")
    t_blur = create_annotated_tile(proc_dict["blurred"], "Gaussian Blur (7x7)")
    t_edge = create_annotated_tile(proc_dict["vertical_edges"], "Vertical Edges (Sobel-X)")

    top_row = np.hstack([t_orig, t_gray])
    bottom_row = np.hstack([t_blur, t_edge])
    grid = np.vstack([top_row, bottom_row])

    return cv2.resize(grid, (display_width, display_height), interpolation=cv2.INTER_AREA)


def run_pipeline(
    video_path: str,
    initial_view: str = "detected",
    min_area: float = 800.0,
    max_area: float = 50000.0,
    min_ratio: float = 2.0,
    max_ratio: float = 6.0,
    max_tilt: float = 15.0,
):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Error: Specified path is not a valid file: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    if cv2 is None or np is None:
        print("Error: Required dependencies (OpenCV/NumPy) are not installed.", file=sys.stderr)
        print("Please install requirements using: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: OpenCV could not open video file: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, 1280 / max(orig_w, 1), 720 / max(orig_h, 1))
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    preprocessor = Preprocessor(blur_ksize=(7, 7), sobel_ksize=3)
    plate_finder = PlateFinder(
        min_area=min_area,
        max_area=max_area,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        max_tilt=max_tilt,
    )

    current_view = initial_view
    paused = False

    print("=== License Plate Detector Controls ===")
    print(" [d] Detection View (Annotated Plates)")
    print(" [g] 2x2 Multi-Stage Grid View")
    print(" [1] Original BGR View")
    print(" [2] Grayscale View")
    print(" [3] Gaussian Blurred View")
    print(" [4] Vertical Edges (Sobel-X) View")
    print(" [SPACE] Pause / Resume")
    print(" [q] Quit")
    print("=======================================")

    window_name = "License Plate Detector"
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
                candidates = plate_finder.find_possible_plates(frame)

            if current_view == "grid":
                display_img = build_grid_view(proc, disp_w, disp_h)
            elif current_view == "gray":
                display_img = create_annotated_tile(proc["gray"], "Grayscale")
            elif current_view == "blurred":
                display_img = create_annotated_tile(proc["blurred"], "Gaussian Blur")
            elif current_view == "edges":
                display_img = create_annotated_tile(proc["vertical_edges"], "Vertical Edges (Sobel-X)")
            elif current_view == "original":
                display_img = create_annotated_tile(proc["original"], "Original Stream")
            else:
                annotated = frame.copy()
                for cand in candidates:
                    cv2.drawContours(annotated, [cand["box"]], 0, (0, 255, 0), 3)
                    rect = cand["rect"]
                    w_b, h_b = rect[1]
                    ratio = (max(w_b, h_b) / min(w_b, h_b)) if min(w_b, h_b) > 0 else 0
                    label = f"Plate Candidate (r={ratio:.1f})"
                    lx = max(10, int(rect[0][0] - max(w_b, h_b) / 2))
                    ly = max(24, int(rect[0][1] - min(w_b, h_b) / 2 - 8))
                    cv2.putText(
                        annotated,
                        label,
                        (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                display_img = create_annotated_tile(annotated, f"Detected Plates ({len(candidates)})")

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(delay if not paused else 50) & 0xFF

            if key == ord('q'):
                print(f"\nPlayback stopped by user at frame {frame_count}.")
                break
            elif key == ord(' '):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"[{status}] Frame {frame_count}")
            elif key == ord('d'):
                current_view = "detected"
            elif key == ord('g'):
                current_view = "grid"
            elif key == ord('1') or key == ord('o'):
                current_view = "original"
            elif key == ord('2'):
                current_view = "gray"
            elif key == ord('3'):
                current_view = "blurred"
            elif key == ord('4') or key == ord('e'):
                current_view = "edges"

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        initial_view=args.view,
        min_area=args.min_area,
        max_area=args.max_area,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        max_tilt=args.max_tilt,
    )


if __name__ == "__main__":
    main()
