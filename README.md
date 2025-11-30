# Phase 1 - Autonomy Perception

Advanced Driver Assistance System (ADAS) with 100+ features for perception, detection, and monitoring.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Quick Start

```bash
# Clone repository
git clone https://github.com/Sherin-SEF-AI/Phase1-Autonomy-Perception.git
cd Phase1-Autonomy-Perception

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application with all features
python3 run_all_features.py
```

## Features

### Core Detection & Tracking
- **Object Detection** - YOLOv8-based detection (people, vehicles, bikes)
- **Multi-Object Tracking** - Kalman filter-based tracking with unique IDs
- **Lane Detection** - Real-time lane line detection
- **Collision Warning** - Forward collision warnings
- **Safe Distance Monitoring** - Tailgating detection

### Traffic & Road Analysis
- **Traffic Sign Recognition** - Stop, yield, speed limits
- **Traffic Light Detection** - Red/yellow/green state detection
- **Road Debris Detection** - Obstacle and debris detection
- **Pothole Detection** - Road surface damage detection
- **Parking Space Detection** - Available/occupied space marking

### AI-Powered Features
- **Scene Classification** - Road/highway/urban/parking detection
- **Vehicle Type Classification** - Car/truck/bus/motorcycle
- **Weather Detection** - Clear/rainy/foggy/snowy conditions
- **Driver Attention Monitoring** - Drowsiness & distraction detection
- **Emergency Vehicle Detection** - Police/ambulance/fire truck
- **Night Vision Enhancement** - Low-light image enhancement

### Advanced Analytics
- **Optical Flow Analysis** - Dense motion field visualization
- **Motion Prediction** - Future trajectory prediction
- **Driving Behavior Analysis** - Aggressive/normal/cautious scoring
- **Pedestrian Pose Estimation** - 33-point body keypoints
- **License Plate Detection** - Vehicle plate recognition
- **Object Density Heatmap** - Heat map visualization

### Visualization & Monitoring
- **Statistics Dashboard** - Real-time metrics panel
- **Performance Graphs** - FPS, CPU, object count over time
- **Multi-Camera Display** - Support for 4 cameras
- **Real-time Overlays** - Bounding boxes, tracking IDs, predictions

## Available Versions

| Version | Features | FPS | Best For |
|---------|----------|-----|----------|
| `run_all_features.py` | 20+ | 8-15 | All capabilities showcase |
| `adas_complete_ultra.py` | Custom | 8-30 | Flexible configuration |
| `adas_fast.py` | 6 | 25-35 | Real-time performance |
| `adas-perception.py` | 20 | 20-25 | Standard production use |

### Run with All Features
```bash
python3 run_all_features.py
```

### Run with Custom Selection
```bash
python3 adas_complete_ultra.py
# Feature selection dialog appears - choose your features
```

### Run Speed-Optimized Version
```bash
python3 adas_fast.py
```

## Feature Selection Dialog

The `adas_complete_ultra.py` version includes a pre-launch dialog where you can:

- Choose exactly which features to enable
- See real-time FPS estimates
- Use quick presets (All, Balanced, Speed, None)
- Save/Load configurations as profiles
- View performance impact for each feature

**Quick Presets:**
- **All Features** - Enable everything (8-15 FPS)
- **Balanced** - Good features without heavy ones (18-22 FPS) - Recommended
- **Speed Mode** - Essential features only (25-30 FPS)
- **Disable All** - Core ADAS only (30-40 FPS)

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (i5 or better)
- **Camera**: Webcam or video file

### Python Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
mediapipe>=0.10.0
wxPython>=4.2.0
filterpy>=1.4.5
Pillow>=10.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Documentation

- **START_HERE.txt** - First-time setup guide
- **HOW_TO_RUN_ALL_FEATURES.txt** - Complete guide for all features
- **FEATURE_SELECTION_QUICKSTART.txt** - Feature selection quick reference
- **WHICH_VERSION_TO_RUN.txt** - Version selection guide
- **ALL_RUN_OPTIONS.txt** - All available run options

## Usage Examples

### Example 1: Basic Usage
```bash
# Run with balanced preset (recommended for first-time users)
python3 adas_complete_ultra.py
# Click "Balanced" -> "Start Application"
```

### Example 2: Demo/Presentation
```bash
# Run speed-optimized for smooth playback
python3 adas_fast.py
```

### Example 3: Full Feature Showcase
```bash
# Run with all 100+ features
python3 run_all_features.py
```

### Example 4: Custom Configuration
```bash
# Use feature selection dialog
python3 adas_complete_ultra.py
# Manually select desired features
# Save as profile for reuse
```

## Configuration

### Camera Settings

Edit camera settings in the application:
- Number of cameras: 1-4
- Resolution: 640x480, 1280x720, 1920x1080
- FPS: 15, 30, 60

### Feature Settings

Toggle features at runtime using GUI checkboxes:
- **ULTRA FEATURES** section (orange labels)
- **AI FEATURES** section (green labels)

Or configure before launch using feature selection dialog.

## Performance Tips

For best performance:
- Use **1 camera** instead of 4
- Use **640x480 resolution** for 4x speed boost
- Close background applications
- Use **"Balanced"** or **"Speed Mode"** presets
- Disable heavy features if FPS is low

Heavy features (consider disabling):
- Driver Attention Monitoring (-6 FPS)
- Pedestrian Pose Estimation (-7 FPS)
- Parking Space Detection (-5 FPS)
- Optical Flow Visualization (-5 FPS)

## Project Structure

```
Phase1-Autonomy-Perception/
├── adas_complete_ultra.py      # Main app with feature selection
├── run_all_features.py          # Quick launcher (all features)
├── adas_fast.py                 # Speed-optimized version
├── adas-perception.py           # Standard version
├── ultra_features.py            # Advanced detection features
├── ultra_ai_features.py         # Modern AI features
├── adas_ultra_advanced.py       # Core ultra features
├── ultra_visualization.py       # Visualization components
├── advanced_modules.py          # Analytics modules
├── requirements.txt             # Python dependencies
├── examples/                    # Example scripts
│   ├── basic_usage.py
│   └── advanced_analytics.py
└── docs/                        # Documentation files
    ├── START_HERE.txt
    ├── HOW_TO_RUN_ALL_FEATURES.txt
    └── ...
```

## Testing

Run the test suite:
```bash
python3 test_ultra_features.py
```

Test individual features:
```bash
python3 examples/basic_usage.py
python3 examples/advanced_analytics.py
```

## Troubleshooting

### Application is slow
- Use fewer cameras (1 instead of 4)
- Lower resolution (640x480)
- Disable heavy features
- Try `adas_fast.py`

### Features not working
- Check that all dependencies are installed
- Verify camera is connected
- Check console for error messages

### Camera not detected
- Check camera permissions
- Try different camera index (0, 1, 2, etc.)
- Test with: `python3 -c "import cv2; print(cv2.VideoCapture(0).read())"`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **YOLOv8** by Ultralytics - Object detection
- **MediaPipe** by Google - Pose estimation
- **OpenCV** - Computer vision
- **wxPython** - GUI framework

## Author

**Sherin Joseph Roy**
- Co-Founder & Head of Products, DeepMost AI
- Location: Bangalore, India
- Website: [sherinjosephroy.link](https://sherinjosephroy.link)
- GitHub: [github.com/Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)
- LinkedIn: [linkedin.com/in/sherin-roy-deepmost](https://www.linkedin.com/in/sherin-roy-deepmost)
- X (Twitter): [x.com/SherinSEF](https://x.com/SherinSEF)
- Mastodon: [mastodon.social/@sherinjoesphroy](https://mastodon.social/@sherinjoesphroy)

## Project

**Project**: Phase 1 - Autonomy Perception
**Organization**: DeepMost AI
**Repository**: [github.com/Sherin-SEF-AI/Phase1-Autonomy-Perception](https://github.com/Sherin-SEF-AI/Phase1-Autonomy-Perception)

---

**Built for Advanced Driver Assistance**

*Developed with assistance from Claude Code*
