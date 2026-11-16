"""
src/ocr.py - Optical Character Recognition (OCR) Backends

Defines an abstract Recognizer interface and two concrete implementations:
  1. TesseractRecognizer: Rule/engine-based OCR using Tesseract (single-character mode --psm 10).
  2. KerasCNNRecognizer: Deep learning classification via a custom-trained Keras CNN model.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


# 36 alphanumeric classes: Digits (0-9) followed by Uppercase Latin (A-Z)
ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_TO_CHAR = {i: ch for i, ch in enumerate(ALPHANUMERIC_CHARS)}
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(ALPHANUMERIC_CHARS)}


class Recognizer(ABC):
    """
    Abstract Base Class for isolated character and plate string recognition.
    """

    @abstractmethod
    def recognize_char(self, char_img: Any) -> Tuple[str, float]:
        """
        Classifies a single character patch.

        Args:
            char_img: 2D or 3D NumPy array of the character crop.

        Returns:
            Tuple of (predicted_char_string, confidence_score [0.0 - 1.0])
        """
        pass

    def recognize(self, char_images: List[Any]) -> str:
        """
        Processes an ordered list of character patches and concatenates
        predictions into a complete license plate string.

        Args:
            char_images: List of segmented character patches ordered left-to-right.

        Returns:
            Assembled license plate string (e.g., 'DL01AB1234').
        """
        results = []
        for img in char_images:
            char, _ = self.recognize_char(img)
            results.append(char)
        return "".join(results)

    def recognize_detailed(self, char_images: List[Any]) -> List[Dict[str, Any]]:
        """
        Returns predictions along with per-character confidence scores.
        """
        details = []
        for i, img in enumerate(char_images):
            char, conf = self.recognize_char(img)
            details.append({
                "index": i + 1,
                "char": char,
                "confidence": conf,
            })
        return details


class TesseractRecognizer(Recognizer):
    """
    Backend A: Tesseract OCR configured for single-character recognition.
    Uses --psm 10 (treat image as single character) and an alphanumeric whitelist.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Args:
            tesseract_cmd: Optional path to the tesseract binary.
        """
        self.tesseract_cmd = tesseract_cmd
        self._pytesseract = None
        self._check_installation()

    def _check_installation(self):
        try:
            import pytesseract
            self._pytesseract = pytesseract
            if self.tesseract_cmd:
                self._pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        except ImportError:
            self._pytesseract = None

    def recognize_char(self, char_img: Any) -> Tuple[str, float]:
        if self._pytesseract is None:
            raise RuntimeError(
                "pytesseract is not installed or tesseract binary not found. "
                "Install via: pip install pytesseract, and install the tesseract binary."
            )

        if char_img is None or char_img.size == 0:
            return "", 0.0

        # Preprocess character for Tesseract:
        # 1. Grayscale & Resize with padding to ensure high resolution (~60x80px)
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img.copy()

        # Add white border padding (Tesseract struggles if characters touch image boundaries)
        padded = cv2.copyMakeBorder(
            gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )

        # Invert if text is white on black: Tesseract expects black text on white background
        mean_val = np.mean(padded)
        if mean_val < 127:
            padded = cv2.bitwise_not(padded)

        # Single character mode (--psm 10) with alphanumeric whitelist
        config = (
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "--psm 10"
        )

        try:
            # Extract data with confidence
            data = self._pytesseract.image_to_data(
                padded, config=config, output_type=self._pytesseract.Output.DICT
            )
            text_candidates = []
            for txt, conf in zip(data["text"], data["conf"]):
                clean = txt.strip().upper()
                if clean:
                    confidence = float(conf) / 100.0 if float(conf) > 0 else 0.5
                    text_candidates.append((clean, confidence))

            if text_candidates:
                return text_candidates[0]

            # Fallback to direct string conversion
            raw_text = self._pytesseract.image_to_string(padded, config=config).strip().upper()
            return (raw_text[:1], 0.6) if raw_text else ("?", 0.0)

        except Exception as e:
            return "?", 0.0


class KerasCNNRecognizer(Recognizer):
    """
    Backend B: Custom Convolutional Neural Network trained on alphanumeric glyphs (0-9, A-Z).
    Normalizes character images to 28x28 grayscale tensors for inference.
    """

    def __init__(
        self,
        model_path: str = "models/char_cnn.keras",
        labels: str = ALPHANUMERIC_CHARS,
    ):
        """
        Args:
            model_path: Path to the serialized Keras model file.
            labels: String of 36 ordered class characters.
        """
        self.model_path = model_path
        self.labels = labels
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Warning: Failed to load Keras model from {self.model_path}: {e}", file=sys.stderr)
            self.model = None

    def preprocess_patch(self, char_img: Any, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Transforms raw character crop into a normalized tensor:
          - Grayscale conversion
          - Invert to white text on black background (standard MNIST format)
          - Resize to target_size (28, 28)
          - Normalize intensities to [0.0, 1.0]
          - Expand dims to (1, 28, 28, 1)
        """
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img.copy()

        # Check background intensity: ensure white character strokes on black background
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)

        # Pad to square aspect ratio to avoid distortion during resize
        h, w = gray.shape
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        squared = cv2.copyMakeBorder(
            gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

        resized = cv2.resize(squared, target_size, interpolation=cv2.INTER_AREA)
        tensor = resized.astype("float32") / 255.0
        tensor = np.expand_dims(tensor, axis=(0, -1))
        return tensor

    def recognize_char(self, char_img: Any) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError(
                f"Keras model not loaded. Model file '{self.model_path}' was not found. "
                "Train the model first using: python3 train.py"
            )

        if char_img is None or char_img.size == 0:
            return "", 0.0

        tensor = self.preprocess_patch(char_img)
        preds = self.model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        if class_idx < len(self.labels):
            return self.labels[class_idx], confidence
        return "?", 0.0
