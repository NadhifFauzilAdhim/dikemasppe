"""
Detection Visualizer Module.

Handles all visual rendering of detection results on frames:
- Bounding boxes with class colors
- Labels with confidence scores
- FPS counter
- Detection count summary
- Status overlay
"""

import cv2
import numpy as np
from typing import Optional

from config.settings import Settings
from core.postprocessor import FrameResult, Detection


class DetectionVisualizer:
    """
    Renders detection results on video frames.

    Draws bounding boxes, labels, FPS counter, and detection
    summary overlays with per-class color coding.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Draw all detection visualizations on a frame.

        Args:
            frame: BGR frame to draw on
            result: FrameResult containing detections

        Returns:
            Frame with all visualizations drawn
        """
        output = frame.copy()

        # Draw each detection
        for detection in result.detections:
            self._draw_detection(output, detection)

        # Draw FPS counter
        if self.settings.visualization.show_fps:
            self._draw_fps(output, result.fps, result.inference_time_ms)

        # Draw detection count summary
        if self.settings.visualization.show_count:
            self._draw_count_summary(output, result)

        return output

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        """Draw a single detection (bounding box + label)."""
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        color = self._get_class_color(detection.class_id)
        thickness = self.settings.visualization.bbox_thickness

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents for a modern look
        corner_len = max(15, int(min(x2 - x1, y2 - y1) * 0.15))
        corner_thickness = thickness + 1
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thickness)

        # Draw label
        if self.settings.visualization.show_labels:
            self._draw_label(frame, detection, x1, y1, color)

    def _draw_label(
        self,
        frame: np.ndarray,
        detection: Detection,
        x: int,
        y: int,
        color: tuple
    ) -> None:
        """Draw label with background above the bounding box."""
        # Build label text
        label = detection.class_name
        if self.settings.visualization.show_confidence:
            label += f" {detection.confidence:.0%}"

        font_scale = self.settings.visualization.font_scale
        font_thickness = self.settings.visualization.font_thickness

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self._font, font_scale, font_thickness
        )

        # Label background position (above bounding box)
        label_y1 = max(0, y - text_h - 10)
        label_y2 = y

        # Draw label background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, label_y1), (x + text_w + 8, label_y2), color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw text
        text_color = (0, 0, 0)  # Black text on colored background
        cv2.putText(
            frame, label,
            (x + 4, label_y2 - 4),
            self._font, font_scale, text_color, font_thickness,
            cv2.LINE_AA
        )

    def _draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        inference_ms: float
    ) -> None:
        """Draw FPS and inference time in top-left corner."""
        h, w = frame.shape[:2]

        # Background panel
        panel_h = 60
        panel_w = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(
            frame, fps_text,
            (20, 35), self._font, 0.6, fps_color, 2, cv2.LINE_AA
        )

        # Inference time
        inf_text = f"Inference: {inference_ms:.1f}ms"
        cv2.putText(
            frame, inf_text,
            (20, 58), self._font, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

    def _draw_count_summary(self, frame: np.ndarray, result: FrameResult) -> None:
        """Draw detection count summary in top-right corner."""
        h, w = frame.shape[:2]
        class_counts = result.class_counts

        if not class_counts:
            return

        # Calculate panel dimensions
        line_height = 25
        panel_h = 40 + len(class_counts) * line_height
        panel_w = 200

        x_start = w - panel_w - 10
        y_start = 10

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + panel_w, y_start + panel_h),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        title = f"Detected: {result.detection_count}"
        cv2.putText(
            frame, title,
            (x_start + 10, y_start + 22),
            self._font, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Per-class counts
        y_offset = y_start + 45
        for cls_name, count in sorted(class_counts.items()):
            # Find class_id for color lookup
            cls_id = next(
                (k for k, v in self.settings.class_names.items() if v == cls_name),
                0
            )
            color = self._get_class_color(cls_id)

            # Color indicator dot
            cv2.circle(frame, (x_start + 18, y_offset - 4), 5, color, -1)

            # Class name and count
            text = f"{cls_name}: {count}"
            cv2.putText(
                frame, text,
                (x_start + 30, y_offset),
                self._font, 0.45, (220, 220, 220), 1, cv2.LINE_AA
            )

            y_offset += line_height

    def _get_class_color(self, class_id: int) -> tuple:
        """Get BGR color for a given class ID."""
        return self.settings.visualization.class_colors.get(
            class_id, (255, 255, 255)
        )

    @staticmethod
    def draw_status_banner(
        frame: np.ndarray,
        text: str,
        color: tuple = (0, 128, 255)
    ) -> np.ndarray:
        """
        Draw a status banner at the bottom of the frame.

        Args:
            frame: Frame to draw on
            text: Status text
            color: Banner color (BGR)

        Returns:
            Frame with status banner
        """
        output = frame.copy()
        h, w = output.shape[:2]

        banner_h = 40
        overlay = output.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)

        cv2.putText(
            output, text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        return output
