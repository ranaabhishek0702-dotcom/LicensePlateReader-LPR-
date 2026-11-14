# Real-Time Car License Plate Detector & Reader

An educational step-by-step implementation of an Automatic Number Plate Recognition (ANPR / ALPR) system in Python using OpenCV, NumPy, and Scikit-Image.

---

## 6-Phase Learning Roadmap

This project is developed incrementally across 6 phases:
- **Phase 1 (Current):** Project scaffolding, video streaming basics, and pipeline architecture.
- **Phase 2:** Frame preprocessing (grayscale, noise reduction, and vertical edge detection).
- **Phase 3:** Binarization (Otsu's thresholding) and morphological transformations.
- **Phase 4:** Contour detection, geometric filtering, and plate region-of-interest (ROI) extraction.
- **Phase 5:** Character segmentation within the detected plate.
- **Phase 6:** Optical Character Recognition (OCR) and real-time bounding box/text overlay.

---

## Pipeline Overview (Plain English)

The classical computer vision pipeline detects and reads license plates using the following sequential steps:

1. **Blur & Grayscale (`blur/gray`)**
   - **What it does:** Converts color 3-channel BGR video frames into single-channel grayscale, then applies Gaussian smoothing.
   - **Why:** Color information is not needed to spot rectangular borders or alphanumeric text. Blurring eliminates high-frequency noise and pixel artifacts that would cause false edge detections.

2. **Vertical Edge Extraction (`vertical edges`)**
   - **What it does:** Applies Sobel filtering along the x-axis (horizontal gradient / vertical edges).
   - **Why:** License plates are composed of dense, closely spaced characters and rectangular border lines that generate strong vertical edges compared to the smooth surface of the car body.

3. **Thresholding (`threshold`)**
   - **What it does:** Converts the gradient edge image into a pure binary image (black & white pixels) using Otsu's thresholding.
   - **Why:** Binarization cleanly separates high-contrast edge regions (foreground white pixels) from flat or low-contrast areas (background black pixels) without needing hardcoded pixel cutoff values.

4. **Morphological Transformations (`morphology`)**
   - **What it does:** Applies a morphological **Closing** operation using a wide, rectangular structuring element (kernel).
   - **Why:** Closing bridges the tiny dark gaps between adjacent vertical character strokes, turning the scattered letters into a single, cohesive white rectangular blob.

5. **Contour Extraction (`contours`)**
   - **What it does:** Uses border-following algorithms (`cv2.findContours`) to detect boundaries of all connected white regions in the binary image.
   - **Why:** Provides a list of coordinate polygons representing all candidate objects in the scene.

6. **Validate Plate Shape (`validate plate shape`)**
   - **What it does:** Computes the bounding rectangle and aspect ratio (width / height) and surface area for each contour.
   - **Why:** Filters out non-plate shapes (headlights, bumpers, road signs) by enforcing geometric constraints (e.g., standard license plates have an aspect ratio typically between 2.0 and 5.5 and fall within reasonable pixel area limits).

7. **Character Segmentation (`segment characters`)**
   - **What it does:** Crops the candidate plate region, extracts the brightness/value channel in HSV, applies adaptive thresholding, and finds individual character contours from left to right.
   - **Why:** Isolates each alphanumeric character as its own clean bitmap, eliminating plate frame borders, screws, or shadows.

8. **Optical Character Recognition (`OCR`)**
   - **What it does:** Passes each segmented character (or the cropped plate) into a recognition model (neural network or OCR engine) to predict alphanumeric characters (`0–9`, `A–Z`).
   - **Why:** Converts visual pixel representations into a machine-readable text string for vehicle identification.

---

## Project Structure

```
NumberPlateDetector/
├── README.md               # Project documentation and pipeline guide
├── requirements.txt        # Core dependencies
├── main.py                 # Main execution script (Phase 1: Video playback)
├── src/                    # Source package for modular components
│   └── __init__.py
└── data/                   # Test videos and images
```

---

## Phase 1: What I Should Understand Checklist

Before moving to Phase 2, make sure you can answer and understand:

- [ ] **Video as a sequence of frames:** A video file is simply a stream of discrete image matrices (frames) read sequentially over time.
- [ ] **`cv2.VideoCapture` mechanics:** How OpenCV opens a video file or camera stream, buffers data, and exposes capture properties (FPS, resolution, frame count).
- [ ] **The `ret` boolean flag:** Why `cap.read()` returns two values, what `ret == False` signals (end of file vs. decode failure), and why testing `ret` prevents segmentation faults or `NoneType` errors.
- [ ] **Frame timing with `cv2.waitKey`:** Why `cv2.waitKey(delay)` is required to render GUI windows in OpenCV, how `delay` controls playback speed, and how key event masks (`& 0xFF == ord('q')`) capture keyboard input.
- [ ] **Resource management:** Why `cap.release()` and `cv2.destroyAllWindows()` must be called to free OS video decoding handles and GUI resources.
