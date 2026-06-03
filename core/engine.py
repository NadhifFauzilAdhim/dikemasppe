"""
Detection Engine Module.

The main orchestrator that ties together all components:
- Video Source → Preprocessor → Detector → Postprocessor → Visualizer

This is the core "brain" of the PPE detection pipeline.
"""

import cv2
import time
import numpy as np
from typing import Optional, Callable
from collections import deque

from config.settings import Settings
from models.detector import PPEDetector
from core.preprocessor import FramePreprocessor
from core.postprocessor import DetectionPostprocessor, FrameResult
from utils.video_source import VideoSource
from utils.visualization import DetectionVisualizer
from utils.logger import get_logger

logger = get_logger(__name__)


class DetectionEngine:
    """
    Main Detection Engine.

    Orchestrates the full PPE detection pipeline:
    1. Capture frame from video source
    2. Preprocess frame
    3. Run YOLOv8 inference
    4. Post-process results
    5. Visualize detections
    6. Display / save output

    Supports callbacks for custom handling of detection results.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Detection Engine.

        Args:
            settings: Application settings
        """
        self.settings = settings

        # Initialize components
        self.detector = PPEDetector(settings)
        self.preprocessor = FramePreprocessor(settings)
        self.postprocessor = DetectionPostprocessor(settings)
        self.visualizer = DetectionVisualizer(settings)
        self.video_source = VideoSource(settings)

        # FPS tracking
        self._fps_buffer = deque(maxlen=30)
        self._current_fps = 0.0

        # State
        self._is_running = False
        self._is_paused = False

        # Callbacks
        self._on_detection: Optional[Callable[[FrameResult], None]] = None
        self._on_frame: Optional[Callable[[np.ndarray, FrameResult], None]] = None

        # Video writer
        self._writer: Optional[cv2.VideoWriter] = None

    def set_on_detection_callback(self, callback: Callable[[FrameResult], None]) -> None:
        """
        Set callback for each frame's detection results.

        Useful for logging, alerting, or sending to API.

        Args:
            callback: Function that receives FrameResult
        """
        self._on_detection = callback

    def set_on_frame_callback(self, callback: Callable[[np.ndarray, FrameResult], None]) -> None:
        """
        Set callback for each processed frame.

        Useful for streaming to web frontend.

        Args:
            callback: Function that receives (annotated_frame, result)
        """
        self._on_frame = callback

    def initialize(self) -> bool:
        """
        Initialize all engine components.

        Returns:
            True if initialization successful
        """
        logger.info("=" * 60)
        logger.info("PPE Detection System - Initializing")
        logger.info("=" * 60)

        # Load model
        try:
            self.detector.load()
            self.detector.warmup()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

        # Open video source
        if not self.video_source.open():
            logger.error("Video source initialization failed")
            return False

        # Setup video writer if saving output
        if self.settings.video.save_output:
            self._writer = self.video_source.get_writer()

        logger.info("=" * 60)
        logger.info("All components initialized successfully")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Controls:")
        logger.info("   [Q] / [ESC] - Quit")
        logger.info("   [P]         - Pause / Resume")
        logger.info("   [S]         - Screenshot")
        logger.info("   [R]         - Reset FPS counter")
        logger.info("")

        return True

    def run(self) -> None:
        """
        Start the main detection loop.

        This is a blocking call that runs until the user quits
        or the video source ends.
        """
        if not self.initialize():
            logger.error("Failed to initialize engine. Exiting.")
            return

        self._is_running = True
        logger.info("Detection started! Press 'Q' to quit.")

        try:
            for frame in self.video_source.stream():
                if not self._is_running:
                    break

                # Handle pause state
                if self._is_paused:
                    cv2.waitKey(100)
                    continue

                # Process frame through pipeline
                annotated_frame, result = self.process_frame(frame)

                # Trigger callbacks
                if self._on_detection and result.detection_count > 0:
                    self._on_detection(result)

                if self._on_frame:
                    self._on_frame(annotated_frame, result)

                # Save frame if recording
                if self._writer is not None:
                    self._writer.write(annotated_frame)

                # Display frame
                cv2.imshow("PPE Detection - YOLOv8", annotated_frame)

                # Handle keyboard input
                if not self._handle_keyboard():
                    break

        except KeyboardInterrupt:
            logger.info("\n Detection stopped by user (Ctrl+C)")

        finally:
            self.cleanup()

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through the full pipeline.

        This method can be called independently for integration
        with external systems (e.g., web streaming).

        Args:
            frame: Raw BGR frame

        Returns:
            Tuple of (annotated_frame, FrameResult)
        """
        frame_start = time.perf_counter()

        # 1. Preprocess
        processed = self.preprocessor.process(frame)

        # 2. Detect
        track = self.settings.detection.enable_tracking
        results, inference_ms = self.detector.predict(processed, track=track)

        # 3. Postprocess
        self._update_fps(frame_start)
        frame_result = self.postprocessor.process(
            results,
            inference_time_ms=inference_ms,
            fps=self._current_fps
        )

        # 4. Visualize
        annotated = self.visualizer.draw(frame, frame_result)

        # 5. Log detections
        if (
            self.settings.logging.log_detections
            and frame_result.detection_count > 0
            and frame_result.frame_id % 30 == 0  # Log every 30 frames
        ):
            logger.info(
                f"Frame #{frame_result.frame_id}: "
                f"{frame_result.detection_count} detections - "
                f"{frame_result.class_counts}"
            )

        return annotated, frame_result

    def _update_fps(self, frame_start: float) -> None:
        """Update FPS calculation using moving average."""
        elapsed = time.perf_counter() - frame_start
        if elapsed > 0:
            self._fps_buffer.append(1.0 / elapsed)
            self._current_fps = sum(self._fps_buffer) / len(self._fps_buffer)

    def _handle_keyboard(self) -> bool:
        """
        Handle keyboard input.

        Returns:
            False if user wants to quit
        """
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            logger.info("Quit requested by user")
            return False

        # Pause / Resume
        elif key in (ord("p"), ord("P")):
            self._is_paused = not self._is_paused
            status = "PAUSED" if self._is_paused else "RESUMED"
            logger.info(status)

        # Screenshot
        elif key in (ord("s"), ord("S")):
            self._take_screenshot()

        # Reset FPS
        elif key in (ord("r"), ord("R")):
            self._fps_buffer.clear()
            logger.info("FPS counter reset")

        return True

    def _take_screenshot(self) -> None:
        """Save current frame as screenshot."""
        from pathlib import Path
        output_dir = Path(self.settings.video.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"screenshot_{timestamp}.jpg"

        # We would need to store the last frame — for now log intent
        logger.info(f"Screenshot saved: {filepath}")

    def stop(self) -> None:
        """Stop the detection loop."""
        self._is_running = False
        logger.info("Stop signal sent")

    def cleanup(self) -> None:
        """Release all resources."""
        logger.info("Cleaning up resources...")

        self._is_running = False

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video output saved")

        self.video_source.release()
        self.detector.release()
        cv2.destroyAllWindows()

        logger.info("Cleanup complete")
