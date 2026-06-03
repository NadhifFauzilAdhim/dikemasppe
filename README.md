# 🦺 PPE Detection System - YOLOv8

Real-time Personal Protective Equipment detection using **YOLOv8n (Nano)** with 6 safety classes.

---

## 📌 Detected Classes

| ID | Class Name   | Color   |
|----|-------------|---------|
| 0  | Gloves       | 🟢 Green  |
| 1  | Vest         | 🟠 Orange |
| 2  | Goggles      | 🔵 Cyan   |
| 3  | Helmet       | 🔵 Blue   |
| 4  | Mask         | 🟣 Pink   |
| 5  | Safety Shoe  | 🟡 Yellow |

---

## 📂 Project Structure

```
dikemasppe/
├── config/
│   ├── __init__.py           # Config package
│   ├── settings.py           # Centralized settings (dataclass)
│   └── config.yaml           # YAML configuration (editable)
│
├── core/
│   ├── __init__.py           # Core package
│   ├── engine.py             # Main detection engine (orchestrator)
│   ├── preprocessor.py       # Frame preprocessing (ROI, enhancement)
│   └── postprocessor.py      # Result filtering & structuring
│
├── models/
│   ├── __init__.py           # Models package
│   └── detector.py           # YOLOv8 model wrapper
│
├── utils/
│   ├── __init__.py           # Utils package
│   ├── logger.py             # Colored logging utility
│   ├── video_source.py       # Video capture manager
│   └── visualization.py      # Bounding box & overlay drawing
│
├── yolomodel/
│   └── best.pt               # YOLOv8n trained weights
│
├── outputs/                   # Saved videos, screenshots, logs
├── main.py                    # 🚀 Entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Detection

```bash
# Default webcam detection
python main.py

# From video file
python main.py --source path/to/video.mp4

# From RTSP stream
python main.py --source rtsp://192.168.1.100:554/stream

# GPU with high confidence
python main.py --device cuda --conf 0.6

# Enable object tracking
python main.py --track

# Save output video
python main.py --save

# FP16 half precision (GPU only, faster)
python main.py --device cuda --half
```

### 3. Keyboard Controls

| Key         | Action           |
|-------------|------------------|
| `Q` / `ESC` | Quit             |
| `P`          | Pause / Resume   |
| `S`          | Take Screenshot  |
| `R`          | Reset FPS Counter|

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize settings without touching code:

```yaml
model:
  confidence_threshold: 0.5   # Lower = more detections
  device: "auto"               # "auto", "cpu", "cuda"

video:
  source: "0"                  # Webcam index or path
  frame_width: 1280
  frame_height: 720

detection:
  target_classes: null         # null = all, or [0, 3, 4]
  enable_tracking: false
```

---

## 🏗️ Architecture

```
Video Source → Preprocessor → YOLOv8 Model → Postprocessor → Visualizer → Display
     │                                              │
     └──────── Frame ──────────► Inference ────► Results ────► Annotated Frame
```

Each component is **independent** and **pluggable**:
- Swap video source without changing model code
- Customize visualization without touching detection logic
- Add callbacks for external integrations (API, database, alerts)

---

## 📡 Integration (for Backend/API)

The engine supports **callback hooks** for easy integration:

```python
from config.settings import get_settings
from core.engine import DetectionEngine

settings = get_settings()
engine = DetectionEngine(settings)

# Custom callback - send results to API
def on_detection(result):
    print(f"Detected: {result.class_counts}")
    # Send to FastAPI, database, websocket, etc.

engine.set_on_detection_callback(on_detection)
engine.run()
```

---

## 📝 License

This project is for educational and research purposes.
Human supervision is mandatory for high-risk safety applications.
