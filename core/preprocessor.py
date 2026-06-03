"""
Frame Preprocessor Module.

Handles all frame preprocessing operations before inference:
- Resizing
- Normalization  
- Region of Interest (ROI) cropping
- Frame enhancement
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from config.settings import Settings


class FramePreprocessor:
    """
    Preprocesses video frames before feeding to YOLOv8 model.

    The Ultralytics YOLOv8 pipeline handles most preprocessing internally,
    but this class provides additional preprocessing options for:
    - ROI cropping (focus on specific area)
    - Frame enhancement (brightness, contrast)
    - Frame resizing for display
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._roi: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
        self._enhancement_enabled = False
        self._brightness = 0
        self._contrast = 1.0

    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set Region of Interest for cropped detection."""
        self._roi = (x1, y1, x2, y2)

    def clear_roi(self) -> None:
        """Clear Region of Interest."""
        self._roi = None

    def set_enhancement(
        self,
        enabled: bool = True,
        brightness: int = 0,
        contrast: float = 1.0
    ) -> None:
        """
        Configure frame enhancement.

        Args:
            enabled: Enable/disable enhancement
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
        """
        self._enhancement_enabled = enabled
        self._brightness = np.clip(brightness, -100, 100)
        self._contrast = np.clip(contrast, 0.5, 3.0)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all preprocessing steps to a frame.

        Args:
            frame: Raw BGR frame from video source

        Returns:
            Preprocessed frame ready for inference
        """
        processed = frame.copy()

        # Step 1: Apply ROI cropping
        if self._roi is not None:
            x1, y1, x2, y2 = self._roi
            h, w = processed.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            processed = processed[y1:y2, x1:x2]

        # Step 2: Apply enhancement
        if self._enhancement_enabled:
            processed = self._enhance_frame(processed)

        return processed

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        adjusted = cv2.convertScaleAbs(
            frame,
            alpha=self._contrast,
            beta=self._brightness
        )
        return adjusted

    def resize_for_display(
        self,
        frame: np.ndarray,
        max_width: int = 1280,
        max_height: int = 720
    ) -> np.ndarray:
        """
        Resize frame for display while maintaining aspect ratio.

        Args:
            frame: Frame to resize
            max_width: Maximum display width
            max_height: Maximum display height

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]

        if w <= max_width and h <= max_height:
            return frame

        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
