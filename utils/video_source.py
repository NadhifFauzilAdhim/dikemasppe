"""
Video Source Manager Module.

Handles video capture from various sources:
- Webcam (USB/built-in)
- Video files (MP4, AVI, etc.)
- RTSP streams
- IP cameras
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoSource:
    """
    Manages video capture from various sources.

    Supports webcam, video files, RTSP streams, and IP cameras.
    Provides a clean iterator interface for frame acquisition.
    """

    def __init__(self, settings: Settings):
        """
        Initialize VideoSource.

        Args:
            settings: Application settings containing video configuration
        """
        self.settings = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._source_type = "unknown"
        self._frame_count = 0

    def open(self, source: Optional[str] = None) -> bool:
        """
        Open video source.

        Args:
            source: Override source (webcam index, file path, or RTSP URL).
                    If None, uses settings.video.source.

        Returns:
            True if source opened successfully
        """
        source = source or self.settings.video.source

        # Determine source type
        self._source_type = self._detect_source_type(source)
        logger.info(f"📹 Opening video source: {source} (type: {self._source_type})")

        # Parse webcam index
        if self._source_type == "webcam":
            source = int(source)

        # Open capture
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            logger.error(f"❌ Failed to open video source: {source}")
            return False

        # Configure capture properties
        self._configure_capture()

        self._is_opened = True
        self._frame_count = 0

        # Log source info
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"✅ Video source opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if not self._is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if ret:
            self._frame_count += 1

        return ret, frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.

        Yields:
            BGR frames as numpy arrays

        Usage:
            for frame in video_source.stream():
                # process frame
        """
        if not self._is_opened:
            logger.error("Video source not opened. Call open() first.")
            return

        while True:
            ret, frame = self.read()
            if not ret:
                if self._source_type == "file":
                    logger.info("📼 End of video file reached")
                else:
                    logger.warning("⚠️  Failed to read frame")
                break
            yield frame

    @property
    def is_opened(self) -> bool:
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        """Get source FPS."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        """Get total frame count (only for video files)."""
        if self._cap is None or self._source_type != "file":
            return -1
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_writer(
        self,
        output_path: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> cv2.VideoWriter:
        """
        Create a VideoWriter for saving output.

        Args:
            output_path: Output file path (default from settings)
            fps: Output FPS (default from settings)

        Returns:
            Configured VideoWriter instance
        """
        output_path = output_path or self.settings.video.output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fps = fps or self.settings.video.fps
        w, h = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*self.settings.video.output_codec)

        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        logger.info(f"💾 Video writer created: {output_path} ({w}x{h} @ {fps} FPS)")

        return writer

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._is_opened = False
            logger.info("🗑️  Video source released")

    def _configure_capture(self) -> None:
        """Configure capture properties based on settings."""
        if self._cap is None:
            return

        if self._source_type == "webcam":
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.video.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.video.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.settings.video.fps)

        # Set buffer size (reduces latency for live sources)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.settings.video.buffer_size)

    @staticmethod
    def _detect_source_type(source: str) -> str:
        """Detect the type of video source."""
        # Webcam (integer index)
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return "webcam"

        source_lower = source.lower()

        # RTSP / HTTP streams
        if source_lower.startswith(("rtsp://", "http://", "https://")):
            return "stream"

        # Video file
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
        if Path(source).suffix.lower() in video_extensions:
            return "file"

        return "unknown"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()
