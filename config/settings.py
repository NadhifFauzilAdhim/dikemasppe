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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    class_colors: Dict[int, tuple] = field(default_factory=lambda: {
        0: (0, 200, 0),      # Hardhat         - Green
        1: (255, 165, 0),    # Mask            - Orange
        2: (0, 0, 255),      # NO-Hardhat      - Red
        3: (0, 0, 200),      # NO-Mask         - Dark Red
        4: (0, 0, 180),      # NO-Safety Vest  - Red
        5: (255, 255, 0),    # Person          - Cyan
        6: (0, 255, 255),    # Safety Cone     - Yellow
        7: (0, 255, 128),    # Safety Vest     - Green
        8: (200, 100, 50),   # machinery       - Blue
        9: (180, 130, 70),   # vehicle         - Steel Blue
    })


@dataclass
class DetectionConfig:
    """Detection & Filtering Configuration."""
    # Minimum area (pixels^2) to consider a detection valid
    min_bbox_area: int = 500
    # Classes to detect (None = all classes
    target_classes: Optional[List[int]] = None
    # Enable tracking
    enable_tracking: bool = False
    tracker_type: str = "bytetrack"  # "bytetrack" or "botsort"


@dataclass
class LoggingConfig:
    """Logging Configuration."""
    level: str = "INFO"  
    log_to_file: bool = False
    log_file_path: str = str(OUTPUTS_DIR / "detection.log")
    log_detections: bool = True


@dataclass
class ApiConfig:
    """API Upload Configuration for violation reporting."""
    enabled: bool = False
    base_url: str = "http://localhost:8000"
    endpoint: str = "/api/v1/violations"
    api_key: str = ""
    camera_id: str = "CAM-001"
    timeout: int = 10
    max_retries: int = 3
    cooldown_seconds: int = 30
    save_local: bool = True
    capture_dir: str = str(OUTPUTS_DIR / "violations")
    max_saved_images: int = 100     # Auto-delete old images if count exceeds this
    # Image compression settings
    jpeg_quality: int = 60          # JPEG quality (1-100), lower = smaller file
    max_image_width: int = 800      # Max width in pixels (0 = no resize)
    max_image_height: int = 600     # Max height in pixels (0 = no resize)


@dataclass
class Settings:
    """Master Settings - Combines all config sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    # PPE Class Names
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "machinery",
        9: "vehicle",
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

        if "api" in config_data:
            for key, value in config_data["api"].items():
                if hasattr(settings.api, key):
                    setattr(settings.api, key, value)

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
        settings = Settings.from_yaml(config_path)
    else:
        default_config = CONFIG_DIR / "config.yaml"
        if default_config.exists():
            settings = Settings.from_yaml(str(default_config))
        else:
            settings = Settings()
            
    # Override with environment variables if present
    if os.getenv("API_BASE_URL"):
        settings.api.base_url = os.getenv("API_BASE_URL")
    if os.getenv("API_ENDPOINT"):
        settings.api.endpoint = os.getenv("API_ENDPOINT")
    if os.getenv("API_KEY"):
        settings.api.api_key = os.getenv("API_KEY")
    if os.getenv("CAMERA_ID"):
        settings.api.camera_id = os.getenv("CAMERA_ID")
    if os.getenv("VIDEO_SOURCE"):
        settings.video.source = os.getenv("VIDEO_SOURCE")

    return settings
