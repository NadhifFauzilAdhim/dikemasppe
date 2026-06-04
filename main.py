"""
PPE Detection System - Main Entry Point.

YOLOv8n-based Personal Protective Equipment detection
with real-time video processing.

Usage:
    # Default (webcam, default settings):
    python main.py

    # With custom config:
    python main.py --config config/config.yaml

    # With video file:
    python main.py --source path/to/video.mp4

    # With RTSP stream:
    python main.py --source rtsp://192.168.1.100:554/stream

    # GPU with half precision:
    python main.py --device cuda --half

    # With object tracking:
    python main.py --track
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, Settings
from core.engine import DetectionEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PPE Detection System - YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Webcam with defaults
  python main.py --source video.mp4           # Video file
  python main.py --source rtsp://ip/stream    # RTSP stream
  python main.py --conf 0.6 --device cuda     # Custom confidence, GPU
  python main.py --track                      # Enable object tracking
  python main.py --save                       # Save output video
        """
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source: webcam index (0), video path, or RTSP URL"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to YOLOv8 model weights (.pt or .onnx)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Inference device"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Confidence threshold (0.0 - 1.0)"
    )
    parser.add_argument(
        "--iou", type=float, default=None,
        help="IoU threshold for NMS (0.0 - 1.0)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Inference image size (e.g., 640)"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use FP16 half precision inference (GPU only)"
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable object tracking (ByteTrack)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without display window (headless mode)"
    )

    return parser.parse_args()


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply CLI argument overrides to settings."""
    if args.source is not None:
        settings.video.source = args.source

    if args.weights is not None:
        settings.model.weights_path = args.weights

    if args.device is not None:
        settings.model.device = args.device

    if args.conf is not None:
        settings.model.confidence_threshold = args.conf

    if args.iou is not None:
        settings.model.iou_threshold = args.iou

    if args.imgsz is not None:
        settings.model.input_size = args.imgsz

    if args.half:
        settings.model.half_precision = True

    if args.track:
        settings.detection.enable_tracking = True

    if args.save:
        settings.video.save_output = True

    return settings


def print_banner() -> None:
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   PPE DETECTION SYSTEM by Nadhif Fauzil A                ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                ║
║                                                          ║
║   Model     : YOLOv8 PPE                                 ║
║   Classes   : Hardhat, Mask, NO-Hardhat, NO-Mask,        ║
║               NO-Safety Vest, Person, Safety Cone,       ║
║               Safety Vest, machinery, vehicle            ║
║   Framework : Ultralytics + OpenCV                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    print_banner()

    # Parse CLI arguments
    args = parse_args()

    # Load settingsclear
    
    settings = get_settings(args.config)
    settings = apply_cli_overrides(settings, args)

    # Log configuration
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Model weights: {settings.model.weights_path}")
    logger.info(f"Video source: {settings.video.source}")
    logger.info(f"Device: {settings.model.device}")
    logger.info(f"Confidence: {settings.model.confidence_threshold}")
    logger.info(f"Tracking: {'ON' if settings.detection.enable_tracking else 'OFF'}")

    # Create and run engine
    engine = DetectionEngine(settings)

    # Example: Add a detection callback (optional)
    def on_detection(result):
        """Custom callback - fires when detections are found."""
        # You can log to database, send alerts, etc.
        pass

    engine.set_on_detection_callback(on_detection)

    # Run the detection pipeline
    engine.run()


if __name__ == "__main__":
    main()
