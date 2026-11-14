"""
src/plate_finder.py - Vehicle License Plate Localization

Implements candidate plate localization:
1. Preprocessing: Blur(7x7) -> Grayscale -> Sobel-X -> Otsu -> Morphological Close (22x3)
2. External contour extraction
3. Geometric filtering (Area, Aspect Ratio, Tilt Angle)
"""

from typing import Tuple, List, Dict, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class PlateFinder:
    """
    Localizes candidate license plate regions from an input image
    using classical computer vision techniques.
    """

    def __init__(
        self,
        min_area: float = 800.0,
        max_area: float = 50000.0,
        min_ratio: float = 2.0,
        max_ratio: float = 6.0,
        max_tilt: float = 15.0,
    ):
        """
        Args:
            min_area: Minimum bounding box pixel area to accept.
            max_area: Maximum bounding box pixel area to accept.
            min_ratio: Minimum aspect ratio (width / height) typical of license plates.
            max_ratio: Maximum aspect ratio (width / height) typical of license plates.
            max_tilt: Maximum allowed deviation (in degrees) from horizontal.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_tilt = max_tilt

        # Structuring element for morphological closing (wide rectangular kernel)
        if cv2 is not None:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
        else:
            self.morph_kernel = None

    def preprocess_steps(self, img: Any) -> Dict[str, Any]:
        """
        Executes each preprocessing step sequentially and returns all intermediate images.
        Sequence:
          1. Gaussian blur (7x7)
          2. Grayscale conversion
          3. Sobel in X direction (vertical edges)
          4. Otsu's thresholding
          5. Morphological CLOSE (22x3 kernel)
        """
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV (cv2) and NumPy are required.")

        # 1. Gaussian blur (7x7)
        blurred = cv2.GaussianBlur(img, (7, 7), 0)

        # 2. Grayscale conversion
        if len(blurred.shape) == 3:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        else:
            gray = blurred.copy()

        # 3. Sobel in X direction (vertical edges)
        # Using 16-bit signed integer to prevent negative gradient clipping
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, dx=1, dy=0, ksize=3)
        grad_x = cv2.convertScaleAbs(sobel_x)

        # 4. Otsu thresholding (automatic binarization)
        _, thresh = cv2.threshold(
            grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 5. Morphological CLOSE with 22x3 rectangular kernel
        if self.morph_kernel is None:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morph_kernel)

        return {
            "blur": blurred,
            "gray": gray,
            "sobel": grad_x,
            "threshold": thresh,
            "closed": closed,
        }

    def preprocess(self, img: Any) -> Any:
        """
        Runs the full preprocessing sequence and returns the final binary image.
        """
        return self.preprocess_steps(img)["closed"]

    def extract_contours(self, binary: Any) -> List[Any]:
        """
        Finds external contours from the preprocessed binary image.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required.")

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    def validate_ratio(self, rect: Tuple) -> bool:
        """
        Validates whether a minimum area bounding box (from cv2.minAreaRect)
        matches standard license plate geometry:
          - Tilt <= max_tilt (default 15 degrees)
          - min_area <= area <= max_area
          - min_ratio <= aspect_ratio <= max_ratio

        Args:
            rect: Tuple of ((center_x, center_y), (width, height), angle)
                  returned by cv2.minAreaRect()
        """
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV (cv2) and NumPy are required.")

        (center, (w, h), angle) = rect

        # Reject degenerate boxes with 0 dimension
        if w <= 0 or h <= 0:
            return False

        # Calculate 4 box corners to determine exact orientation and dimensions
        box = cv2.boxPoints(rect)

        # Find side vectors
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        len1 = float(np.hypot(edge1[0], edge1[1]))
        len2 = float(np.hypot(edge2[0], edge2[1]))

        if len1 <= 0 or len2 <= 0:
            return False

        # License plates are wider than they are tall
        # Identify the longer edge as 'width' and shorter as 'height'
        if len1 >= len2:
            long_edge = edge1
            width = len1
            height = len2
        else:
            long_edge = edge2
            width = len2
            height = len1

        # Calculate tilt of the long edge relative to horizontal
        # angle = arctan(|dy| / |dx|)
        tilt_rad = np.arctan2(abs(long_edge[1]), abs(long_edge[0]))
        tilt_deg = float(np.degrees(tilt_rad))

        # Check 1: Reject tilt over max_tilt (15 degrees)
        if tilt_deg > self.max_tilt:
            return False

        # Check 2: Surface area constraint
        area = width * height
        if not (self.min_area <= area <= self.max_area):
            return False

        # Check 3: Aspect ratio constraint (width / height)
        aspect_ratio = width / height
        if not (self.min_ratio <= aspect_ratio <= self.max_ratio):
            return False

        return True

    def find_possible_plates(self, img: Any) -> List[Dict[str, Any]]:
        """
        Full detection pipeline on a single image.
        Returns a list of validated candidate plate dictionaries:
          - 'contour': raw OpenCV contour
          - 'rect': cv2.minAreaRect output
          - 'box': integer 4-point corner polygon
        """
        steps = self.preprocess_steps(img)
        binary = steps["closed"]
        contours = self.extract_contours(binary)

        candidates = []
        for cnt in contours:
            # Need at least 4 points to form a polygon
            if len(cnt) < 4:
                continue

            rect = cv2.minAreaRect(cnt)
            if self.validate_ratio(rect):
                box = np.intp(cv2.boxPoints(rect))
                candidates.append({
                    "contour": cnt,
                    "rect": rect,
                    "box": box,
                })

        return candidates
