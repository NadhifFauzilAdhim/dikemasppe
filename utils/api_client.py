"""
API Client Module.

Handles uploading violation data (capture image + metadata)
to the dashboard backend server via HTTP POST.

Uploads run in a background thread to avoid blocking
the main detection loop.
"""

import io
import cv2
import time
import json
import threading
import numpy as np
import requests
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================
# Violation Classes (NO-* prefixed)
# ============================================
VIOLATION_CLASS_IDS = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest

VIOLATION_LABELS = {
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
}


@dataclass
class ViolationPayload:
    """Structured violation data ready for API upload."""
    timestamp: str
    camera_id: str
    violation_type: str          # e.g. "NO-Hardhat"
    violation_class_id: int
    confidence: float
    bbox: Dict[str, int]         # {x1, y1, x2, y2}
    person_count: int
    all_detections: List[dict]
    frame_id: int
    inference_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class ViolationUploader:
    """
    Handles violation detection, image capture, and API upload.

    When a violation (NO-*) is detected, this module:
    1. Captures the annotated frame as JPEG
    2. Packages violation metadata as JSON
    3. Sends both to the backend via multipart POST
    4. Uses a background thread + queue for non-blocking upload

    Includes cooldown logic to avoid flooding the server
    with duplicate violations from consecutive frames.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # API config
        self._api_url = settings.api.base_url.rstrip("/") + settings.api.endpoint
        self._api_key = settings.api.api_key
        self._camera_id = settings.api.camera_id
        self._timeout = settings.api.timeout
        self._max_retries = settings.api.max_retries
        self._enabled = settings.api.enabled

        # Cooldown: prevent duplicate uploads for same violation type
        self._cooldown_seconds = settings.api.cooldown_seconds
        self._last_upload_time: Dict[str, float] = {}

        # Background upload queue
        self._upload_queue: Queue = Queue(maxsize=50)
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Image compression settings
        self._jpeg_quality = max(1, min(100, settings.api.jpeg_quality))
        self._max_width = settings.api.max_image_width
        self._max_height = settings.api.max_image_height

        # Stats
        self._upload_count = 0
        self._error_count = 0

        # Local save directory for violation captures
        self._capture_dir = Path(settings.api.capture_dir)
        if settings.api.save_local:
            self._capture_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start background upload worker thread."""
        if not self._enabled:
            logger.info("API upload is disabled in config")
            return

        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._upload_worker,
            daemon=True,
            name="ViolationUploader"
        )
        self._worker_thread.start()
        logger.info(f"Violation uploader started -> {self._api_url}")

    def stop(self) -> None:
        """Stop the background upload worker."""
        self._is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        logger.info(
            f"Violation uploader stopped. "
            f"Uploaded: {self._upload_count}, Errors: {self._error_count}"
        )

    def check_and_upload(self, frame: np.ndarray, frame_result) -> None:
        """
        Check if frame contains violations and queue upload.

        Args:
            frame: The annotated BGR frame (with bounding boxes drawn)
            frame_result: FrameResult from postprocessor
        """
        if not self._enabled:
            return

        # Find violation detections in this frame
        violations = [
            det for det in frame_result.detections
            if det.class_id in VIOLATION_CLASS_IDS
        ]

        if not violations:
            return

        # Count persons in frame
        person_count = sum(
            1 for det in frame_result.detections if det.class_id == 5
        )

        # Process each violation type (group by type for cooldown)
        violation_types_in_frame = set()
        for det in violations:
            vtype = det.class_name
            if vtype in violation_types_in_frame:
                continue
            violation_types_in_frame.add(vtype)

            # Check cooldown
            now = time.time()
            last_time = self._last_upload_time.get(vtype, 0)
            if now - last_time < self._cooldown_seconds:
                continue

            self._last_upload_time[vtype] = now

            # Compress and encode frame as JPEG bytes
            jpeg_bytes = self._compress_image(frame)
            if jpeg_bytes is None:
                logger.warning("Failed to compress frame")
                continue

            # Save locally if enabled
            if self.settings.api.save_local:
                self._save_local(jpeg_bytes, vtype, frame_result.frame_id)

            # Build payload
            payload = ViolationPayload(
                timestamp=datetime.now().isoformat(),
                camera_id=self._camera_id,
                violation_type=vtype,
                violation_class_id=det.class_id,
                confidence=round(det.confidence, 4),
                bbox={
                    "x1": int(det.bbox[0]),
                    "y1": int(det.bbox[1]),
                    "x2": int(det.bbox[2]),
                    "y2": int(det.bbox[3]),
                },
                person_count=person_count,
                all_detections=[d.to_dict() for d in frame_result.detections],
                frame_id=frame_result.frame_id,
                inference_time_ms=round(frame_result.inference_time_ms, 2),
            )

            # Queue for background upload
            try:
                self._upload_queue.put_nowait((payload, jpeg_bytes))
                logger.info(
                    f"[VIOLATION] {vtype} (conf: {det.confidence:.2f}) "
                    f"- queued for upload"
                )
            except Exception:
                logger.warning("Upload queue full, dropping violation")

    def _upload_worker(self) -> None:
        """Background worker that processes the upload queue."""
        while self._is_running:
            try:
                payload, jpeg_bytes = self._upload_queue.get(timeout=1)
                self._send_to_server(payload, jpeg_bytes)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Upload worker error: {e}")
                self._error_count += 1

    def _send_to_server(self, payload: ViolationPayload, jpeg_bytes: bytes) -> bool:
        """
        Send violation data to the server via multipart POST.

        Args:
            payload: Violation metadata
            jpeg_bytes: JPEG-encoded capture image

        Returns:
            True if upload successful
        """
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Generate filename for the image
        safe_timestamp = payload.timestamp.replace(":", "-").replace(".", "-")
        filename = f"violation_{payload.violation_type}_{safe_timestamp}.jpg"

        # Multipart form data
        files = {
            "image": (filename, jpeg_bytes, "image/jpeg"),
        }
        data = {
            "payload": json.dumps(payload.to_dict()),
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = requests.post(
                    self._api_url,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=self._timeout,
                )

                if response.status_code in (200, 201):
                    self._upload_count += 1
                    logger.info(
                        f"[UPLOAD OK] {payload.violation_type} "
                        f"uploaded successfully (attempt {attempt})"
                    )
                    return True
                else:
                    logger.warning(
                        f"[UPLOAD FAIL] Server returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )

            except requests.ConnectionError:
                logger.warning(
                    f"[UPLOAD FAIL] Connection error (attempt {attempt}/{self._max_retries})"
                )
            except requests.Timeout:
                logger.warning(
                    f"[UPLOAD FAIL] Timeout (attempt {attempt}/{self._max_retries})"
                )
            except Exception as e:
                logger.error(f"[UPLOAD FAIL] Unexpected error: {e}")
                break

            # Wait before retry
            if attempt < self._max_retries:
                time.sleep(2 * attempt)

        self._error_count += 1
        return False

    def _compress_image(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Compress image by resizing and lowering JPEG quality.

        Steps:
        1. Resize to fit within max_width x max_height (maintain aspect ratio)
        2. Encode as JPEG with configured quality level

        Args:
            frame: BGR image (numpy array)

        Returns:
            Compressed JPEG bytes, or None on failure
        """
        try:
            img = frame.copy()
            h, w = img.shape[:2]

            # Resize if max dimensions are set and image exceeds them
            if self._max_width > 0 and self._max_height > 0:
                if w > self._max_width or h > self._max_height:
                    scale = min(self._max_width / w, self._max_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(
                        img, (new_w, new_h),
                        interpolation=cv2.INTER_AREA
                    )
                    logger.debug(
                        f"Image resized: {w}x{h} -> {new_w}x{new_h}"
                    )

            # Encode with configured JPEG quality
            success, jpeg_buffer = cv2.imencode(
                ".jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )

            if not success:
                return None

            compressed_bytes = jpeg_buffer.tobytes()
            original_size = len(cv2.imencode(".jpg", frame)[1].tobytes())
            compressed_size = len(compressed_bytes)

            logger.debug(
                f"Image compressed: {original_size / 1024:.1f}KB -> "
                f"{compressed_size / 1024:.1f}KB "
                f"({(1 - compressed_size / original_size) * 100:.0f}% reduction)"
            )

            return compressed_bytes

        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return None

    def _save_local(self, jpeg_bytes: bytes, violation_type: str, frame_id: int) -> None:
        """Save violation capture locally as backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{timestamp}_frame{frame_id}.jpg"
        filepath = self._capture_dir / filename

        try:
            filepath.write_bytes(jpeg_bytes)
            logger.debug(f"Violation saved locally: {filepath}")
            
            # Auto-cleanup old images
            self._cleanup_old_images()
        except Exception as e:
            logger.warning(f"Failed to save local capture: {e}")

    def _cleanup_old_images(self) -> None:
        """Keep only the most recent max_saved_images in capture_dir."""
        max_images = getattr(self.settings.api, "max_saved_images", 100)
        if max_images <= 0:
            return
            
        try:
            # Get all jpg files sorted by modification time (oldest first)
            files = sorted(self._capture_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime)
            
            if len(files) > max_images:
                files_to_delete = files[:-max_images]
                for f in files_to_delete:
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete {f}: {e}")
                if files_to_delete:
                    logger.info(f"Auto-cleaned {len(files_to_delete)} old violation images")
        except Exception as e:
            logger.warning(f"Failed to cleanup old images: {e}")

    @property
    def stats(self) -> dict:
        """Get upload statistics."""
        return {
            "enabled": self._enabled,
            "uploads": self._upload_count,
            "errors": self._error_count,
            "queue_size": self._upload_queue.qsize(),
        }
