"""
Detection Postprocessor Module.

Handles all post-processing operations after YOLOv8 inference:
- Filtering detections by confidence, class, area
- Aggregating detection statistics
- Generating structured detection results
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from config.settings import Settings


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) - pixel coordinates
    area: float
    center: tuple  # (cx, cy)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": int(self.bbox[0]),
                "y1": int(self.bbox[1]),
                "x2": int(self.bbox[2]),
                "y2": int(self.bbox[3]),
            },
            "area": int(self.area),
            "center": {"x": int(self.center[0]), "y": int(self.center[1])},
        }


@dataclass
class FrameResult:
    """Detection results for a single frame."""
    frame_id: int
    timestamp: str
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    fps: float = 0.0

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> Dict[str, int]:
        """Count detections per class."""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    @property
    def has_violations(self) -> bool:
        """Check if any required PPE is missing (placeholder logic)."""
        detected_classes = {det.class_name for det in self.detections}
        # This can be customized based on requirements
        return len(detected_classes) == 0

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detection_count": self.detection_count,
            "class_counts": self.class_counts,
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 1),
        }


class DetectionPostprocessor:
    """
    Post-processes raw YOLOv8 detection results.

    Applies filtering, statistics aggregation, and structures
    the results into clean Detection objects.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._frame_counter = 0

    def process(
        self,
        results,
        inference_time_ms: float = 0.0,
        fps: float = 0.0
    ) -> FrameResult:
        """
        Process raw YOLOv8 results into structured FrameResult.

        Args:
            results: Ultralytics YOLOv8 Results object
            inference_time_ms: Time taken for inference
            fps: Current FPS

        Returns:
            FrameResult with filtered and structured detections
        """
        self._frame_counter += 1
        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]  # Single image result

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Extract detection data
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # Get class name
                    cls_name = self.settings.class_names.get(cls_id, f"class_{cls_id}")

                    # --- Filtering ---

                    # Filter by minimum bounding box area
                    if area < self.settings.detection.min_bbox_area:
                        continue

                    # Filter by target classes
                    if (
                        self.settings.detection.target_classes is not None
                        and cls_id not in self.settings.detection.target_classes
                    ):
                        continue

                    detection = Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        area=float(area),
                        center=(float(center[0]), float(center[1])),
                    )
                    detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        frame_result = FrameResult(
            frame_id=self._frame_counter,
            timestamp=datetime.now().isoformat(),
            detections=detections,
            inference_time_ms=inference_time_ms,
            fps=fps,
        )

        return frame_result

    def reset_counter(self) -> None:
        """Reset the frame counter."""
        self._frame_counter = 0
