"""
PPE Detector Module.

Wrapper around Ultralytics YOLOv8 model for PPE detection.
Handles model loading, device management, and inference.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PPEDetector:
    """
    YOLOv8-based PPE Detection Model.

    Encapsulates model loading, device selection, and inference logic.
    Supports PyTorch (.pt) and ONNX (.onnx) model formats.
    """

    def __init__(self, settings: Settings):
        """
        Initialize PPE Detector.

        Args:
            settings: Application settings containing model configuration
        """
        self.settings = settings
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self._is_loaded = False

    def load(self) -> None:
        """
        Load YOLOv8 model and configure device.

        Raises:
            FileNotFoundError: If model weights file doesn't exist
            RuntimeError: If model loading fails
        """
        weights_path = Path(self.settings.model.weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}\n"
                f"Please place your model weights at: {weights_path.absolute()}"
            )

        # Resolve device
        self.device = self._resolve_device(self.settings.model.device)
        logger.info(f"Using device: {self.device}")

        # Load model
        logger.info(f"Loading model: {weights_path.name}")
        try:
            self.model = YOLO(str(weights_path))
            self._is_loaded = True
            logger.info(f"Model loaded successfully")
            logger.info(f"   Classes: {list(self.settings.class_names.values())}")
            logger.info(f"   Input size: {self.settings.model.input_size}x{self.settings.model.input_size}")
            logger.info(f"   Confidence threshold: {self.settings.model.confidence_threshold}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, frame: np.ndarray, track: bool = False):
        """
        Run inference on a single frame.

        Args:
            frame: BGR frame (numpy array)
            track: Enable object tracking

        Returns:
            Tuple of (results, inference_time_ms)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Common inference parameters
        params = {
            "source": frame,
            "imgsz": self.settings.model.input_size,
            "conf": self.settings.model.confidence_threshold,
            "iou": self.settings.model.iou_threshold,
            "max_det": self.settings.model.max_detections,
            "device": self.device,
            "half": self.settings.model.half_precision,
            "verbose": False,
        }

        # Run inference (with or without tracking)
        if track and self.settings.detection.enable_tracking:
            params["tracker"] = f"{self.settings.detection.tracker_type}.yaml"
            params["persist"] = True
            results = self.model.track(**params)
        else:
            results = self.model.predict(**params)

        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        return results, inference_time

    def warmup(self, n: int = 3) -> None:
        """
        Warm up the model with dummy inference.

        This helps optimize GPU memory allocation and CUDA kernels
        for more consistent inference times.

        Args:
            n: Number of warmup iterations
        """
        if not self._is_loaded:
            return

        logger.info(f"Warming up model ({n} iterations)...")
        dummy = np.zeros(
            (self.settings.model.input_size, self.settings.model.input_size, 3),
            dtype=np.uint8
        )

        for _ in range(n):
            self.model.predict(
                source=dummy,
                imgsz=self.settings.model.input_size,
                device=self.device,
                verbose=False,
            )

        logger.info("Warmup complete")

    def get_model_info(self) -> dict:
        """Get model metadata and configuration info."""
        info = {
            "weights": self.settings.model.weights_path,
            "device": self.device,
            "input_size": self.settings.model.input_size,
            "confidence_threshold": self.settings.model.confidence_threshold,
            "iou_threshold": self.settings.model.iou_threshold,
            "half_precision": self.settings.model.half_precision,
            "num_classes": len(self.settings.class_names),
            "class_names": self.settings.class_names,
            "is_loaded": self._is_loaded,
        }

        if self._is_loaded and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"

        return info

    @staticmethod
    def _resolve_device(device_config: str) -> str:
        """
        Resolve the device string to use for inference.

        Args:
            device_config: Device configuration ("auto", "cpu", "cuda", "cuda:0")

        Returns:
            Resolved device string
        """
        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {gpu_name}")
                return device
            else:
                logger.warning("No CUDA GPU detected, falling back to CPU")
                return "cpu"

        return device_config

    def release(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model resources released")
