"""
src/preprocessor.py - Phase 2: Frame Preprocessing & Vertical Edge Extraction

Performs color conversion, noise suppression, and vertical edge isolation
tailored for detecting vehicle license plates.
"""

from typing import Tuple, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None



class Preprocessor:
    """
    Handles initial image transformations to prepare video frames
    for license plate localization.
    """

    def __init__(self, blur_ksize: Tuple[int, int] = (5, 5), sobel_ksize: int = 3):
        """
        Args:
            blur_ksize: Kernel dimensions for Gaussian Blur. Must be odd positive integers.
            sobel_ksize: Aperture size for the Sobel operator. Must be 1, 3, 5, or 7.
        """
        self.blur_ksize = blur_ksize
        self.sobel_ksize = sobel_ksize

    def to_grayscale(self, frame: Any) -> Any:
        """
        Converts a 3-channel BGR image to single-channel 8-bit grayscale.
        Formula: Y = 0.299*R + 0.587*G + 0.114*B
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, gray: Any) -> Any:
        """
        Applies 2D Gaussian smoothing to attenuate high-frequency noise
        and sensor grain before numerical gradient calculation.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        return cv2.GaussianBlur(gray, self.blur_ksize, 0)

    def extract_vertical_edges(self, blurred: Any) -> Any:
        """
        Computes horizontal gradients (vertical edges) via the Sobel operator.
        
        Why dx=1, dy=0?
        License plate characters and the plate's vertical borders exhibit sharp
        vertical contrast shifts, while horizontal lines (e.g. bumpers, grills)
        are suppressed.
        
        Why CV_16S?
        Pixel intensity transitions can be positive (dark-to-bright) or negative
        (bright-to-dark). CV_16S stores signed values so negative derivatives
        are not truncated to 0. convertScaleAbs computes the absolute value and
        scales back to uint8.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")

        # Compute signed gradient in x-direction (vertical edges)
        grad_x = cv2.Sobel(
            blurred,
            ddepth=cv2.CV_16S,
            dx=1,
            dy=0,
            ksize=self.sobel_ksize
        )

        # Convert back to uint8 using absolute value: |grad_x|
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        return abs_grad_x

    def process(self, frame: Any) -> Dict[str, Any]:
        """
        Runs the full Phase 2 preprocessing pipeline on a single frame.

        Returns:
            Dictionary containing:
            - 'original': Input BGR frame
            - 'gray': Grayscale representation
            - 'blurred': Gaussian smoothed grayscale
            - 'vertical_edges': 8-bit vertical gradient magnitude
        """
        gray = self.to_grayscale(frame)
        blurred = self.apply_gaussian_blur(gray)
        vertical_edges = self.extract_vertical_edges(blurred)

        return {
            "original": frame,
            "gray": gray,
            "blurred": blurred,
            "vertical_edges": vertical_edges,
        }
