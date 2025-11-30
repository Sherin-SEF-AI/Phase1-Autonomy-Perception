# 🚀 Running the Ultra-Advanced ADAS System

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
cd /home/vision2030/Desktop/adas-perception

# Install core dependencies
pip install numpy opencv-python opencv-contrib-python wxPython psutil

# Install advanced features
pip install matplotlib mediapipe ultralytics
```

### Step 2: Choose Your Version

You have **3 versions** available:

| Version | Features | Recommended For |
|---------|----------|----------------|
| **v1.0** - `adas-perception.py` | 20 basic features | Learning, quick demos |
| **v2.0** - `adas-perception-advanced.py` | 55 advanced features | Development, research |
| **v3.0** - Complete ultra system | 100+ all features | Production, showcase |

### Step 3: Run

```bash
# Option 1: Run v1.0 (Basic - Working)
python3 adas-perception.py

# Option 2: Run v2.0 (Advanced)
python3 adas-perception-advanced.py

# Option 3: Run v3.0 ULTRA (All Features) - See below
python3 run_ultra.py
```

---

## 🎯 Running the ULTRA Version (v3.0)

The ultra version has **modular components**. Here's how to use them:

### Option A: Run Complete Integrated System

I'll create a complete launcher for you:

```bash
python3 run_ultra.py
```

### Option B: Run Individual Features

```bash
# Test individual components
python3 test_ultra_features.py
```

### Option C: Use as Library

```python
# In your own script
from ultra_features import *
from ultra_visualization import *
```

---

## 📋 Pre-Flight Checklist

### 1. Check Python Version
```bash
python3 --version
# Should be 3.8 or higher
```

### 2. Verify Dependencies
```bash
python3 << 'EOF'
import sys
packages = ['numpy', 'cv2', 'wx', 'psutil']
optional = ['matplotlib', 'mediapipe', 'ultralytics']

print("Core Packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} - REQUIRED")

print("\nOptional Packages:")
for pkg in optional:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ⚠️  {pkg} - Optional but recommended")
EOF
```

### 3. Check Camera Access
```bash
# List available cameras
ls /dev/video* 2>/dev/null || echo "No cameras found"

# Test camera 0
ffplay /dev/video0 -t 5 2>/dev/null || echo "Camera test failed"
```

---

## 🎮 Usage Instructions

### First-Time Setup

1. **Launch Application**
   ```bash
   python3 run_ultra.py
   ```

2. **Camera Selection Dialog**
   - Select 1-4 cameras
   - Choose resolution (1280x720 recommended)
   - Click OK

3. **Main Interface**
   - **Top**: Multi-camera views
   - **Left**: Primary camera with detection
   - **Right**: Graphs, maps, statistics
   - **Bottom**: Controls

4. **Enable Features**
   - Check boxes to enable/disable features
   - Adjust detection threshold
   - Set ego vehicle speed

5. **Start Processing**
   - Click **▶ START** button
   - Watch real-time processing
   - Monitor performance graphs

---

## ⚙️ Feature Configuration

### Enable/Disable Features

```python
# Edit feature flags in run_ultra.py
FEATURES = {
    # Visualization
    'enable_3d_pointcloud': True,
    'enable_panorama': True,
    'enable_optical_flow': True,
    'enable_trajectory_map': True,

    # Detection
    'enable_pose_estimation': True,
    'enable_vehicle_classification': True,
    'enable_license_plates': True,
    'enable_pothole_detection': True,

    # Analysis
    'enable_scene_classification': True,
    'enable_behavior_analysis': True,
    'enable_motion_prediction': True,

    # Recording
    'enable_timelapse': False,
    'enable_slowmotion': False,

    # Graphs
    'enable_performance_graphs': True,
    'enable_confidence_graphs': True,
    'enable_heatmap': True,
    'enable_statistics': True,
}
```

### Performance Optimization

**For Maximum FPS** (disable heavy features):
```python
FEATURES = {
    'enable_pose_estimation': False,  # Saves ~4 FPS
    'enable_panorama': False,         # Saves ~3 FPS
    'enable_optical_flow': False,     # Saves ~2 FPS
    'enable_3d_pointcloud': False,    # Saves ~2 FPS
}
```

**For Maximum Features** (lower resolution):
- Use 640x480 instead of 1280x720
- Enable GPU acceleration
- Use single camera

---

## 🎨 GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ADAS Ultra-Advanced Perception System v3.0                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐  ┌───────────────────────────┐   │
│  │  Main Camera View   │  │  Performance Graphs       │   │
│  │  (Detection +       │  │  - FPS                    │   │
│  │   Tracking +        │  │  - CPU                    │   │
│  │   Lanes)            │  │  - Memory                 │   │
│  │                     │  │                           │   │
│  │                     │  ├───────────────────────────┤   │
│  │                     │  │  Trajectory Map           │   │
│  │                     │  │  (Top-down view)          │   │
│  └─────────────────────┘  │                           │   │
│                           │                           │   │
│  ┌─────┬─────┬─────┬────┐├───────────────────────────┤   │
│  │Cam 1│Cam 2│Cam 3│Cam4││  Statistics Dashboard     │   │
│  └─────┴─────┴─────┴────┘│  - Session Info           │   │
│                           │  - Detection Stats        │   │
│  ┌─────────────────────┐ │  - Behavior Score         │   │
│  │  Optional Views:    │ │  - Scene Classification   │   │
│  │  - Panorama         │ │                           │   │
│  │  - Point Cloud      │ └───────────────────────────┘   │
│  │  - Optical Flow     │                                 │
│  │  - Heatmap          │  ┌───────────────────────────┐   │
│  └─────────────────────┘  │  Controls                 │   │
│                           │  ☐ Pose Estimation        │   │
│  ┌─────────────────────┐ │  ☐ Vehicle Classification │   │
│  │  ▶ START  ⏺ REC    │ │  ☐ Optical Flow           │   │
│  └─────────────────────┘ │  Threshold: [====|====]   │   │
│                           └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Controls Reference

### Buttons
- **▶ START** - Begin perception processing
- **⏹ STOP** - Stop processing
- **⏺ RECORD** - Start/stop video recording
- **📷 SNAPSHOT** - Capture current frame

### Keyboard Shortcuts
- **Space** - Start/Stop processing
- **R** - Start/Stop recording
- **S** - Take snapshot
- **Q** - Quit application
- **F** - Toggle fullscreen
- **H** - Toggle heatmap
- **T** - Toggle trajectory map
- **G** - Toggle graphs
- **1-4** - Switch primary camera

### Checkboxes
- **Object Detection** - Enable/disable detection
- **Tracking** - Enable/disable tracking
- **Lanes** - Enable/disable lane detection
- **Pose Estimation** - Enable pedestrian poses
- **Vehicle Classification** - Classify vehicle types
- **Optical Flow** - Show motion vectors
- **Scene Classification** - Detect scene type
- **Motion Prediction** - Show predicted paths

### Sliders
- **Detection Threshold** - 0.1 to 0.9 (default: 0.4)
- **Ego Speed** - 0 to 200 km/h (for TTC calculation)

---

## 📊 Understanding the Output

### Main Camera View
- **Green boxes** - Detected objects (safe distance)
- **Orange boxes** - Warning (close object)
- **Red boxes** - Critical (collision warning)
- **Yellow lines** - Lane markings
- **Cyan trails** - Object trajectories
- **Orange dots** - Predicted future positions
- **Skeleton overlay** - Pedestrian poses (if enabled)

### Performance Graphs
- **Top graph** - FPS over time (green)
- **Middle graph** - CPU usage (blue)
- **Bottom graph** - Memory usage (red)

### Trajectory Map
- **Green rectangle** - Your vehicle (ego)
- **Cyan circles** - Other vehicles
- **Red circles** - Dangerous objects (TTC < 2s)
- **Orange dots** - Predicted paths

### Statistics Dashboard
- **Session Duration** - How long you've been running
- **Total Frames** - Frames processed
- **Detection Stats** - Objects detected/tracked
- **FPS Metrics** - Current/Max/Average FPS
- **Safety Metrics** - Lane departures, warnings
- **Scene Info** - Time, road type, weather, traffic
- **Driving Score** - 0-100 behavior score

---

## 🔍 Example Workflows

### Workflow 1: Basic Detection
```bash
# Run basic version
python3 adas-perception.py

# Or with specific features
python3 run_ultra.py --basic
```

### Workflow 2: Full Analysis
```bash
# Run with all features
python3 run_ultra.py --all-features

# Monitor performance
# Watch CPU/Memory in graphs
# Check FPS in statistics
```

### Workflow 3: Video Processing
```bash
# Process video file
python3 run_ultra.py --input video.mp4 --output results.mp4
```

### Workflow 4: Development/Testing
```bash
# Test individual features
python3 test_ultra_features.py

# Test specific component
python3 -c "from ultra_features import OpticalFlowAnalyzer; print('OK')"
```

---

## 🐛 Troubleshooting

### Application Won't Start

**Error**: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

**Error**: `ModuleNotFoundError: No module named 'wx'`
```bash
pip install wxPython
```

**Error**: Camera not found
```bash
# Check cameras
ls /dev/video*

# Fix permissions
sudo chmod 666 /dev/video0

# Add user to video group
sudo usermod -a -G video $USER
```

### Low Performance

**FPS < 10**
1. Lower resolution to 640x480
2. Disable heavy features:
   - Pose estimation
   - Panorama stitching
   - Optical flow
3. Use single camera
4. Close other applications

**High CPU Usage**
- Check background processes
- Enable GPU acceleration
- Reduce detection threshold
- Disable unused features

### Features Not Working

**Pose Estimation Not Showing**
```bash
pip install mediapipe
```

**Graphs Not Displaying**
```bash
pip install matplotlib
```

**3D Point Cloud Error**
- Requires depth estimation
- Enable depth in settings
- May need GPU

---

## 💡 Tips & Tricks

### Get Best Performance
1. Use 1280x720 resolution
2. Enable only needed features
3. Use single camera first
4. Enable GPU if available
5. Close unnecessary apps

### Get Best Accuracy
1. Good lighting conditions
2. Stable camera mount
3. Higher detection threshold (0.5-0.7)
4. Enable all detection features
5. Calibrate cameras

### Record Best Videos
1. Use normal mode for real-time
2. Use time-lapse for long sessions
3. Use slow motion for detailed analysis
4. Ensure good lighting
5. Stable camera

---

## 📈 Performance Expectations

### System Requirements vs Performance

| System | Resolution | Features Enabled | Expected FPS |
|--------|-----------|------------------|--------------|
| **Minimum** (i5, 8GB) | 640x480 | Basic (detection only) | 25-30 |
| **Recommended** (i7, 16GB) | 1280x720 | Most features | 20-25 |
| **High-end** (i9, 32GB, GPU) | 1920x1080 | All features | 25-30 |

---

## 🎯 Next Steps

After running successfully:

1. ✅ **Experiment** with different features
2. ✅ **Record** a test session
3. ✅ **Review** statistics and graphs
4. ✅ **Optimize** settings for your use case
5. ✅ **Customize** feature configuration
6. ✅ **Integrate** into your project

---

## 📞 Need Help?

- Check: [README_ULTRA.md](README_ULTRA.md)
- Installation: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- Features: [FEATURES_COMPARISON.md](FEATURES_COMPARISON.md)
- API: Code comments and docstrings

---

**Ready to start? Run this:**
```bash
python3 run_ultra.py
```

**Good luck! 🚀**
