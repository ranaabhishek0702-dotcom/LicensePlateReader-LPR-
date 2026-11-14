# Real-Time Automatic Number Plate Recognition (ANPR)

A modular, real-time Computer Vision system for vehicle license plate detection and character recognition built with Python and OpenCV.

---

## Overview

This repository provides an end-to-end Automatic Number Plate Recognition (ANPR / ALPR) pipeline. It combines classical image processing (spatial gradient analysis, dynamic thresholding, and morphological operations) with geometric contour validation to isolate vehicle license plates from real-time video streams and still images.

---

## System Architecture

The pipeline processes input frames through sequential image processing stages:

```
[ Input Frame (BGR) ]
          │
          ▼
[ 1. Grayscale & Noise Reduction ]  --> cv2.cvtColor, cv2.GaussianBlur (7x7)
          │
          ▼
[ 2. Vertical Edge Extraction ]    --> Sobel-X operator (dx=1, dy=0)
          │
          ▼
[ 3. Dynamic Binarization ]        --> Otsu's Thresholding
          │
          ▼
[ 4. Morphological Consolidation ] --> Morphological Closing (22x3 rectangular kernel)
          │
          ▼
[ 5. Contour Extraction ]          --> cv2.findContours (RETR_EXTERNAL)
          │
          ▼
[ 6. Geometric Validation ]        --> cv2.minAreaRect (Area, Aspect Ratio, Tilt Filter)
          │
          ▼
[ 7. Candidate Plate Extraction ]  --> Normalized Plate ROI for Character Segmentation
```

### Key Technical Decisions
- **Vertical Edge Isolation (Sobel-X):** Exploits the dense vertical character strokes on vehicle plates while suppressing horizontal vehicle lines (bumpers, grilles, hood lines).
- **Signed 16-Bit Derivative Space (`CV_16S`):** Preserves negative spatial gradients (dark-to-bright and bright-to-dark transitions) prior to absolute scaling.
- **Otsu's Thresholding:** Dynamically computes optimal binarization thresholds based on the image histogram to handle changing lighting and shadow conditions.
- **Asymmetric Morphological Closing ($22 \times 3$):** Fuses adjacent character strokes horizontally into a single cohesive rectangular bounding contour without bleeding into adjacent bumper structures.
- **Rotated Minimum Area Bounding (`minAreaRect`):** Computes accurate dimensions and orientation angles, rejecting false positives based on aspect ratio, surface area, and tilt limits.

---

## Project Structure

```
.
├── main.py                 # Real-time video stream runner & multi-view visualizer
├── debug_plate_finder.py   # Diagnostics tool exporting intermediate processing stages
├── requirements.txt        # Package dependencies
├── src/                    # Core library
│   ├── __init__.py
│   ├── plate_finder.py     # Plate localization, morphology, and geometric validation
│   └── preprocessor.py     # Grayscale, Gaussian smoothing, and gradient operations
└── data/                   # Input media (videos, sample frames)
```

---

## Getting Started

### Prerequisites
- Python 3.9 or higher

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ranaabhishek0702-dotcom/LicensePlateReader-LPR-.git
cd LicensePlateReader-LPR-

# 2. Set up a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## Usage

### 1. Real-Time Video Processing
Run the live video pipeline on any recorded video file or stream:

```bash
python3 main.py path/to/video.mp4
```

#### Interactive Keyboard Controls:
| Key | Action |
| :---: | :--- |
| `g` | 2x2 multi-stage grid view (Original, Gray, Blur, Vertical Edges) |
| `1` | Original BGR stream |
| `2` | Grayscale stream |
| `3` | Gaussian blurred stream |
| `4` | Vertical edges (Sobel-X) stream |
| `SPACE` | Pause / resume playback |
| `q` | Exit application |

### 2. Single-Image Diagnostics & Stage Visualization
To analyze a single image and inspect every intermediate image processing step:

```bash
python3 debug_plate_finder.py path/to/car.jpg --output output/
```

This exports 7 sequential diagnostic artifacts to the output directory:
1. `01_blur.jpg`: Gaussian smoothed frame ($7 \times 7$)
2. `02_gray.jpg`: Grayscale luminance channel
3. `03_sobel.jpg`: Vertical edge gradient magnitude (Sobel-X)
4. `04_threshold.jpg`: Otsu threshold binary mask
5. `05_closed.jpg`: Morphological closing ($22 \times 3$ kernel)
6. `06_all_contours.jpg`: All extracted external contours
7. `07_validated_contours.jpg`: Validated plate candidates annotated with geometry metrics

---

## Configuration & Tuning Parameters

Plate detection thresholds can be configured when initializing `PlateFinder`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `min_area` | `800` | Minimum bounding box pixel area |
| `max_area` | `50000` | Maximum bounding box pixel area |
| `min_ratio` | `2.0` | Minimum aspect ratio (width / height) |
| `max_ratio` | `6.0` | Maximum aspect ratio (width / height) |
| `max_tilt` | `15.0` | Maximum allowed angle deviation (degrees) from horizontal |

---

## License

MIT License.
