# 🚀 ADAS Ultra - Quick Reference Card

## ⚡ How to Run

```bash
cd /home/vision2030/Desktop/adas-perception

# Option 1: Run original working version (v1.0)
python3 adas-perception.py

# Option 2: Run with ultra launcher
python3 run_ultra.py

# Option 3: Test all ultra features
python3 test_ultra_features.py
```

---

## 📦 What You Have

### **3 Complete Versions**

| File | Version | Features | When to Use |
|------|---------|----------|-------------|
| `adas-perception.py` | v1.0 | 20 basic | ✅ **START HERE** - Working now! |
| `adas-perception-advanced.py` | v2.0 | 55 advanced | Development & research |
| Ultra modules | v3.0 | 100+ all | Import as library |

### **Ultra Feature Modules** (Use as Library)

```python
# Import and use ultra features
from ultra_features import *
from ultra_visualization import *
from adas_ultra_advanced import *
```

---

## 🎯 Choose Your Path

### **Path 1: Quick Start (Recommended)**
```bash
# Just run the working application
python3 adas-perception.py
```
✅ Works immediately
✅ All basic features
✅ Professional GUI
✅ Perfect for demos

### **Path 2: Test Ultra Features**
```bash
# Test all advanced modules
python3 test_ultra_features.py
```
✅ Verify all features work
✅ See example outputs
✅ Learn how to use modules

### **Path 3: Use Ultra as Library**
```python
# In your own script
from ultra_features import OpticalFlowAnalyzer

analyzer = OpticalFlowAnalyzer()
flow = analyzer.calculate(frame)
visualization = analyzer.visualize(flow, frame.shape)
```
✅ Full control
✅ Custom integration
✅ All advanced features

---

## 🔧 Installation

### Minimum (to run v1.0)
```bash
pip install numpy opencv-python wxPython psutil
```

### Recommended (for all features)
```bash
pip install numpy opencv-python wxPython psutil matplotlib mediapipe ultralytics
```

### Full Install
```bash
pip install -r requirements_ultra.txt
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **RUNNING_GUIDE.md** | ⭐ **Complete running instructions** |
| **README_ULTRA.md** | All features documentation |
| **INSTALLATION_GUIDE.md** | Platform-specific setup |
| **FEATURES_COMPARISON.md** | Version comparison |
| **PROJECT_SUMMARY.md** | Overview & achievements |

---

## 🎮 Quick Controls

### When Running
- **Space** - Start/Stop
- **R** - Record
- **Q** - Quit
- **S** - Snapshot
- **1-4** - Switch camera

### Sliders
- **Detection Threshold**: 0.1 - 0.9
- **Ego Speed**: 0 - 200 km/h

---

## 💡 Feature Availability Matrix

| Feature | v1.0 | v2.0 | v3.0 Ultra Modules |
|---------|------|------|--------------------|
| Object Detection | ✅ | ✅ | ✅ |
| Tracking | ✅ | ✅ | ✅ + Kalman |
| Lane Detection | ✅ | ✅ | ✅ |
| Collision Warning | ✅ | ✅ | ✅ |
| **3D Point Cloud** | ❌ | ❌ | ✅ |
| **Panorama 360°** | ❌ | ❌ | ✅ |
| **Pose Estimation** | ❌ | ❌ | ✅ (33 points) |
| **Optical Flow** | ❌ | ❌ | ✅ |
| **Scene Classification** | ❌ | ❌ | ✅ |
| **Real-time Graphs** | ❌ | ❌ | ✅ |
| **Behavior Analysis** | ❌ | ✅ Basic | ✅ Complete |

---

## 🔍 Example: Use Ultra Features

### Example 1: Optical Flow
```python
from ultra_features import OpticalFlowAnalyzer
import cv2

analyzer = OpticalFlowAnalyzer()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate optical flow
    flow_data = analyzer.calculate(frame)

    if flow_data:
        # Visualize
        flow_vis = analyzer.visualize(flow_data, frame.shape)
        cv2.imshow('Optical Flow', flow_vis)

        # Get dominant motion
        motion_x, motion_y = flow_data['dominant_motion']
        print(f"Motion: X={motion_x:.2f}, Y={motion_y:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Scene Classification
```python
from adas_ultra_advanced import SceneClassifier
import cv2

classifier = SceneClassifier()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Classify scene
    context = classifier.classify(frame, [])

    # Display info
    info = f"Time: {context.time_of_day.name}"
    info += f" | Road: {context.road_type.name}"
    info += f" | Traffic: {context.traffic_density.name}"
    info += f" | Visibility: {context.visibility_score:.2f}"

    cv2.putText(frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Scene Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 3: Vehicle Classification
```python
from ultra_features import VehicleTypeClassifier
import cv2

classifier = VehicleTypeClassifier()

# Assume you have a vehicle bbox from detection
vehicle_bbox = (100, 200, 300, 400)  # x1, y1, x2, y2

frame = cv2.imread('frame.jpg')
vehicle_type = classifier.classify(vehicle_bbox, frame)

print(f"Vehicle Type: {vehicle_type}")
# Output: SEDAN, SUV, TRUCK, BUS, MOTORCYCLE, etc.
```

---

## 🐛 Quick Fixes

### Problem: Camera not found
```bash
ls /dev/video*
sudo chmod 666 /dev/video0
```

### Problem: Module not found
```bash
pip install <module-name>
```

### Problem: Low FPS
```python
# In code, disable heavy features:
config['enable_pose_estimation'] = False  # Saves 4 FPS
config['enable_panorama'] = False         # Saves 3 FPS
config['enable_optical_flow'] = False     # Saves 2 FPS
```

---

## 📊 Performance Guide

### Get 30 FPS
- Use 640x480 resolution
- Enable only detection + tracking
- Single camera
- Basic features only

### Get 20-25 FPS with Advanced Features
- Use 1280x720 resolution
- Enable most features
- 1-2 cameras
- Disable pose estimation

### Use All 100+ Features
- Expect 12-15 FPS
- Use GPU acceleration
- Or process video offline
- 1080p resolution possible

---

## ✅ Recommended Workflow

### Day 1: Get Started
```bash
python3 adas-perception.py
```
Play with basic features, understand the system

### Day 2: Test Ultra Features
```bash
python3 test_ultra_features.py
```
See all advanced features in action

### Day 3: Build Custom App
```python
# my_app.py
from ultra_features import *
# Build your custom integration
```

---

## 🎯 Success Checklist

- [ ] Dependencies installed
- [ ] Can run `python3 adas-perception.py`
- [ ] Camera working
- [ ] Tested basic features
- [ ] Ran `test_ultra_features.py`
- [ ] Explored ultra modules
- [ ] Read documentation
- [ ] Created custom script
- [ ] Achieved good FPS
- [ ] Understanding all features

---

## 🆘 Need Help?

1. **Read**: RUNNING_GUIDE.md
2. **Check**: Test script output
3. **Verify**: Dependencies installed
4. **Test**: Camera access
5. **Try**: Lower resolution
6. **Review**: Error messages

---

## 🚀 You're Ready!

**Start with this command:**
```bash
python3 adas-perception.py
```

**Then explore:**
```bash
python3 test_ultra_features.py
```

**Happy Autonomous Driving! 🚗💨**
