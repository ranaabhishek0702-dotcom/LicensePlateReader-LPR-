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

try:
    from skimage import measure
except ImportError:
    measure = None



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

    def clean_plate(self, plate_img: Any) -> Tuple[Any, Optional[List[int]]]:
        """
        Refines a candidate plate crop by finding its principal content area.
        Applies adaptive Gaussian thresholding, finds the largest contour,
        and re-checks its aspect ratio and area.

        Args:
            plate_img: Cropped candidate plate image (BGR or Grayscale).

        Returns:
            Tuple of (cleaned_plate_img, [x, y, w, h] bounding box)
        """
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV (cv2) and NumPy are required.")

        if plate_img is None or plate_img.size == 0:
            return plate_img, None

        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Adaptive Gaussian thresholding to isolate plate content from background
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = int(np.argmax(areas))
            max_cnt = contours[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            area = w * h
            aspect_ratio = (w / float(h)) if h > 0 else 0

            # Re-check that the inner box adheres to plate geometry
            # Using broader bounds to accommodate tight interior text borders
            if (self.min_area * 0.2 <= area <= self.max_area) and (
                self.min_ratio * 0.7 <= aspect_ratio <= self.max_ratio * 1.3
            ):
                cleaned = plate_img[y : y + h, x : x + w]
                if cleaned.size > 0:
                    return cleaned, [x, y, w, h]

        return plate_img, None

    def segment_chars(
        self,
        plate_img: Any,
        fixed_width: int = 400,
        min_chars: int = 4,
        max_chars: int = 12,
        char_min_aspect: float = 0.1,
        char_max_aspect: float = 1.0,
        min_solidity: float = 0.15,
        min_height_ratio: float = 0.30,
        max_height_ratio: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Segments individual alphanumeric characters from a cleaned license plate:
          1. Extracts V (Value) channel from HSV color space.
          2. Applies adaptive Gaussian thresholding and inverts (white text on black).
          3. Normalizes image width to fixed_width (default 400px).
          4. Performs connected-component labeling via skimage.measure.label.
          5. Filters components using aspect ratio, solidity, and height ratio checks.
          6. Computes convex hulls and draws them onto a candidate mask.
          7. Extracts characters sorted horizontally from left to right.

        Args:
            plate_img: Cropped plate image.
            fixed_width: Canonical width for consistent character scale.
            min_chars: Configurable lower bound for expected plate character count.
            max_chars: Configurable upper bound for expected plate character count.
            char_min_aspect: Minimum character width/height ratio.
            char_max_aspect: Maximum character width/height ratio.
            min_solidity: Minimum ratio of contour area to bounding box area.
            min_height_ratio: Minimum character height relative to plate height.
            max_height_ratio: Maximum character height relative to plate height.

        Returns:
            Dictionary with:
              - 'characters': list of dicts [{'patch': image, 'binary': mask, 'bbox': (x, y, w, h)}]
              - 'count': number of extracted characters
              - 'is_valid_count': boolean flag (min_chars <= count <= max_chars)
              - 'v_channel': V channel image
              - 'thresh': Inverted adaptive threshold binary image
              - 'char_mask': Convex hull candidates mask
              - 'plate_resized': Resized plate image
        """
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV (cv2) and NumPy are required.")

        if plate_img is None or plate_img.size == 0:
            return {
                "characters": [],
                "count": 0,
                "is_valid_count": False,
                "v_channel": None,
                "thresh": None,
                "char_mask": None,
                "plate_resized": None,
            }

        # 1. Extract V (Value) channel from HSV
        if len(plate_img.shape) == 3:
            hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
        else:
            v_channel = plate_img.copy()

        # 2. Adaptive Gaussian thresholding & inversion (white characters on dark background)
        thresh = cv2.adaptiveThreshold(
            v_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        thresh = cv2.bitwise_not(thresh)

        # 3. Canonical resizing to fixed_width (preserving aspect ratio)
        orig_h, orig_w = v_channel.shape[:2]
        aspect = fixed_width / float(orig_w) if orig_w > 0 else 1.0
        new_h = max(10, int(orig_h * aspect))

        if len(plate_img.shape) == 3:
            plate_resized = cv2.resize(plate_img, (fixed_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            plate_resized = cv2.resize(plate_img, (fixed_width, new_h), interpolation=cv2.INTER_AREA)

        thresh_resized = cv2.resize(thresh, (fixed_width, new_h), interpolation=cv2.INTER_NEAREST)

        # 4. Connected-component labeling (skimage.measure.label)
        if measure is not None:
            labels = measure.label(thresh_resized, background=0)
        else:
            # Fallback if skimage is not installed
            _, labels = cv2.connectedComponents(thresh_resized)

        char_candidates = np.zeros(thresh_resized.shape, dtype=np.uint8)
        plate_h, plate_w = thresh_resized.shape

        # 5. Filter components using rule-based criteria & compute convex hulls
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == 0:
                continue

            component_mask = np.zeros(thresh_resized.shape, dtype=np.uint8)
            component_mask[labels == label] = 255

            cnts, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not cnts:
                continue

            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            box_area = w * h

            if box_area == 0 or h == 0:
                continue

            aspect_ratio = w / float(h)
            solidity = area / float(box_area)
            height_ratio = h / float(plate_h)

            # Apply character filter criteria
            if (
                char_min_aspect <= aspect_ratio <= char_max_aspect
                and solidity >= min_solidity
                and min_height_ratio <= height_ratio <= max_height_ratio
            ):
                hull = cv2.convexHull(c)
                cv2.drawContours(char_candidates, [hull], -1, 255, -1)

        # 6. Extract character contours from convex-hulled mask and sort left-to-right
        char_cnts, _ = cv2.findContours(
            char_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        characters = []
        if char_cnts:
            boxes = [cv2.boundingRect(c) for c in char_cnts]
            # Sort left-to-right by x-coordinate: key = box[0]
            sorted_chars = sorted(zip(char_cnts, boxes), key=lambda item: item[1][0])

            for cnt, (x, y, w, h) in sorted_chars:
                char_patch = plate_resized[y : y + h, x : x + w]
                char_binary = thresh_resized[y : y + h, x : x + w]
                characters.append({
                    "patch": char_patch,
                    "binary": char_binary,
                    "bbox": (x, y, w, h),
                })

        count = len(characters)
        is_valid_count = (min_chars <= count <= max_chars)

        return {
            "characters": characters,
            "count": count,
            "is_valid_count": is_valid_count,
            "v_channel": v_channel,
            "thresh": thresh_resized,
            "char_mask": char_candidates,
            "plate_resized": plate_resized,
        }
