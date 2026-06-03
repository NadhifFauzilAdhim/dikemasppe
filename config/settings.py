"""
Configuration Settings for PPE Detection System.

Centralized configuration management using dataclass.
All settings can be overridden via config.yaml or environment variables.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


# ============================================
# Base Paths
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "yolomodel"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR / "config"


@dataclass
class ModelConfig:
    """YOLOv8 Model Configuration."""
    weights_path: str = str(WEIGHTS_DIR / "best.pt")
    input_size: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"
    half_precision: bool = False  # FP16 inference


@dataclass
class VideoConfig:
    """Video Source Configuration."""
    source: str = "0"  # "0" = webcam, or path to video/RTSP URL
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    buffer_size: int = 1
    save_output: bool = False
    output_path: str = str(OUTPUTS_DIR / "output.mp4")
    output_codec: str = "mp4v"


@dataclass
class VisualizationConfig:
    """Visualization & Drawing Configuration."""
    show_confidence: bool = True
    show_labels: bool = True
    show_fps: bool = True
    show_count: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    # Warna per class (BGR format)
    class_colors: Dict[int, tuple] = field(default_factory=lambda: {
        0: (0, 255, 128),    # Gloves      - Green
        1: (255, 165, 0),    # Vest        - Orange
        2: (255, 255, 0),    # Goggles     - Cyan
        3: (0, 128, 255),    # Helmet      - Blue
        4: (147, 20, 255),   # Mask        - Pink
        5: (0, 255, 255),    # Safety Shoe - Yellow
    })


@dataclass
class DetectionConfig:
    """Detection & Filtering Configuration."""
    # Minimum area (pixels^2) to consider a detection valid
    min_bbox_area: int = 500
    # Classes to detect (None = all classes)
    target_classes: Optional[List[int]] = None
    # Enable tracking
    enable_tracking: bool = False
    tracker_type: str = "bytetrack"  # "bytetrack" or "botsort"


@dataclass
class LoggingConfig:
    """Logging Configuration."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = False
    log_file_path: str = str(OUTPUTS_DIR / "detection.log")
    log_detections: bool = True


@dataclass
class Settings:
    """Master Settings - Combines all config sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # PPE Class Names
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Gloves",
        1: "Vest",
        2: "Goggles",
        3: "Helmet",
        4: "Mask",
        5: "Safety Shoe",
    })

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from a YAML configuration file."""
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        settings = cls()

        if "model" in config_data:
            for key, value in config_data["model"].items():
                if hasattr(settings.model, key):
                    setattr(settings.model, key, value)

        if "video" in config_data:
            for key, value in config_data["video"].items():
                if hasattr(settings.video, key):
                    setattr(settings.video, key, value)

        if "visualization" in config_data:
            for key, value in config_data["visualization"].items():
                if hasattr(settings.visualization, key):
                    setattr(settings.visualization, key, value)

        if "detection" in config_data:
            for key, value in config_data["detection"].items():
                if hasattr(settings.detection, key):
                    setattr(settings.detection, key, value)

        if "logging" in config_data:
            for key, value in config_data["logging"].items():
                if hasattr(settings.logging, key):
                    setattr(settings.logging, key, value)

        if "class_names" in config_data:
            settings.class_names = {
                int(k): v for k, v in config_data["class_names"].items()
            }

        return settings


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Factory function to get Settings instance.

    Priority:
    1. Custom YAML config path (if provided)
    2. Default config.yaml in config directory
    3. Default Settings values
    """
    if config_path and os.path.exists(config_path):
        return Settings.from_yaml(config_path)

    default_config = CONFIG_DIR / "config.yaml"
    if default_config.exists():
        return Settings.from_yaml(str(default_config))

    return Settings()
