# ADAS Perception System - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)
```bash
cd /home/vision2030/Desktop/adas-perception
pip install numpy opencv-python wxPython psutil
```

### Step 2: Run Application (30 seconds)
```bash
python3 adas-perception.py
```

### Step 3: Configure (1 minute)
1. Camera selection dialog appears
2. Select your camera(s)
3. Choose resolution (1280x720 recommended)
4. Click OK

### Step 4: Start Perception (30 seconds)
1. Click **▶ START** button
2. Watch the magic happen!

---

## 🎮 Controls

| Button | Function |
|--------|----------|
| ▶ START | Begin perception processing |
| ⏹ STOP | Stop processing |
| ⏺ RECORD | Start/stop video recording |

## 🎛️ Settings

- **Object Detection**: Toggle detection on/off
- **Object Tracking**: Enable trajectory tracking
- **Lane Detection**: Detect road lanes
- **Bird's Eye View**: Top-down perspective
- **Detection Threshold**: Adjust sensitivity (0.1-0.9)
- **Ego Speed**: Set vehicle speed for TTC

---

## 📊 Dashboard Metrics

| Metric | Description |
|--------|-------------|
| **FPS** | Frames per second |
| **Processing** | Time per frame (ms) |
| **Detections** | Objects detected |
| **Tracked** | Objects being tracked |
| **Lane Status** | Lane detection status |
| **Offset** | Distance from lane center |
| **Closest Object** | Distance to nearest object |
| **Risk Level** | Collision risk assessment |

---

## ⚠️ Alert Levels

- **INFO** 🟢 - No threats
- **WARNING** 🟡 - Object approaching
- **DANGER** 🟠 - Close object (<3s TTC)
- **CRITICAL** 🔴 - Imminent collision (<1.5s TTC)

---

## 🎥 Video Processing

### Open Video File
1. Menu → File → Open Video
2. Select MP4/AVI/MKV file
3. Processing begins automatically

### Record Session
1. Start perception
2. Click ⏺ RECORD
3. Video saved as `adas_recording_YYYYMMDD_HHMMSS.mp4`

---

## 🐛 Quick Troubleshooting

### Camera Not Detected
```bash
# List cameras
ls /dev/video*

# Test camera
ffplay /dev/video0

# Fix permissions
sudo chmod 666 /dev/video0
```

### Low FPS
- Lower resolution to 640x480
- Disable unused features
- Close other applications
- Check CPU usage

### Import Errors
```bash
pip install --force-reinstall opencv-python wxPython
```

---

## 📚 Learn More

- **Full Documentation**: [README_ADVANCED.md](README_ADVANCED.md)
- **Installation Help**: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- **Feature Comparison**: [FEATURES_COMPARISON.md](FEATURES_COMPARISON.md)

---

## 🎯 Try These Examples

### Example 1: Basic Usage
```bash
python3 examples/basic_usage.py
```

### Example 2: Advanced Analytics
```bash
python3 examples/advanced_analytics.py
```

---

**That's it! You're ready to go! 🚀**

Happy autonomous driving! 🚗💨
