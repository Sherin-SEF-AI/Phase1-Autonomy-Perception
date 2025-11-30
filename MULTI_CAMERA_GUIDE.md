# 🎥 Multi-Camera Advanced Features Guide

## Complete ADAS Ultra - Multi-Camera System

This guide explains how each camera in the system is utilized with specialized advanced features.

---

## 📹 Camera Assignment & Features

### **PRIMARY CAMERA (Main Display)**
**Focus**: Complete Perception Pipeline

#### Features Enabled:
- ✅ **Object Detection** - YOLOv8 detection of all objects
- ✅ **Multi-Object Tracking** - Persistent tracking with unique IDs
- ✅ **Lane Detection** - Road lane markings
- ✅ **Collision Warning** - Time-to-collision calculation
- ✅ **Scene Classification** - Time/Road/Weather/Traffic analysis
- ✅ **Vehicle Type Classification** - SEDAN/SUV/TRUCK/BUS
- ✅ **Motion Prediction** - Future trajectory prediction (orange dots)
- ✅ **Driving Behavior Analysis** - Score 0-100 with behavior type
- ✅ **Distance Estimation** - Real-world distance to objects

#### Visual Overlays:
```
┌─────────────────────────────────────────────────┐
│ Scene Info Panel (Top-Left)                    │
│ - Time: DAY/NIGHT/DAWN/DUSK                    │
│ - Road: HIGHWAY/URBAN/RURAL/PARKING            │
│ - Traffic: LOW/MEDIUM/HIGH                     │
│ - Condition: DRY/WET/ICY                       │
│ - Visibility: 0.0-1.0                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Detected Objects with Bounding Boxes]        │
│  [Lane Lines in Yellow]                        │
│  [Motion Prediction Dots in Orange]            │
│  [Vehicle Type Labels]                         │
│                                                 │
├─────────────────────────────────────────────────┤
│ Behavior Score (Top-Right)                     │
│ - Behavior: AGGRESSIVE/NORMAL/CAUTIOUS         │
│ - Score: 0-100 (color-coded)                   │
└─────────────────────────────────────────────────┘
```

---

### **CAMERA 0 (Secondary Display 1)**
**Focus**: Pedestrian Safety & Vehicle Identification

#### Specialized Features:
- 🚶 **Pedestrian Pose Estimation** - 33 body keypoints using MediaPipe
  - Standing, Walking, Running, Waving detection
  - Skeleton overlay on detected pedestrians
  - Action classification label

- 🚗 **License Plate Detection** - Plate localization (no OCR)
  - Yellow bounding boxes around plates
  - "PLATE" label for identification

- 👤 **Person Detection** - Focused pedestrian tracking
  - Enhanced person class detection
  - Pose-based behavior analysis

#### Visual Overlays:
```
┌─────────────────────────────────────────────────┐
│ CAM 0: PEDESTRIAN + PLATES                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Person with Skeleton Overlay]                │
│     • 33 keypoints in cyan                     │
│     • Action label (WALKING/RUNNING)           │
│                                                 │
│  [Vehicle with Yellow Plate Box]               │
│     • "PLATE" label                            │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Use Cases:
- Crosswalk monitoring
- Pedestrian intent prediction
- Vehicle identification
- Parking lot surveillance

---

### **CAMERA 1 (Secondary Display 2)**
**Focus**: Motion Analysis & Optical Flow

#### Specialized Features:
- 🌊 **Optical Flow Visualization** - Dense motion field
  - Farneback optical flow algorithm
  - Motion vector arrows (cyan)
  - Magnitude and direction analysis

- 📊 **Motion Statistics** - Real-time motion metrics
  - Dominant motion direction
  - Average magnitude in pixels/frame
  - Motion pattern classification

- 🎯 **Motion Prediction** - Enhanced for this view
  - Future position prediction
  - Trajectory analysis

#### Visual Overlays:
```
┌─────────────────────────────────────────────────┐
│ CAM 1: MOTION ANALYSIS                         │
├─────────────────────────────────────────────────┤
│ Motion: FORWARD                                │
│ Avg: 15.3 px/frame                             │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Motion Vectors as Arrows]                    │
│     → → → → →  (direction and magnitude)       │
│     ↗ ↗ ↗ ↗ ↗                                  │
│     → → → → →                                  │
│                                                 │
│  [Detected Objects with Motion Emphasis]       │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Use Cases:
- Traffic flow analysis
- Camera motion detection
- Speed estimation
- Movement pattern analysis

---

### **CAMERA 2 (Secondary Display 3)**
**Focus**: Road Surface Analysis

#### Specialized Features:
- 🕳️ **Pothole Detection** - Road damage identification
  - HoughCircles-based detection
  - Darkness analysis
  - Radius estimation
  - Purple circular markers

- 🛣️ **Lane Detection** - Enhanced lane marking
  - Yellow lane lines
  - Road boundary detection

- 📊 **Surface Condition** - Road quality metrics
  - Pothole count display
  - Surface roughness (future)

#### Visual Overlays:
```
┌─────────────────────────────────────────────────┐
│ CAM 2: ROAD SURFACE                            │
├─────────────────────────────────────────────────┤
│ Potholes: 3                                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Lane Lines in Yellow]                        │
│                                                 │
│       ○  POTHOLE (purple circle)               │
│                                                 │
│                ○  POTHOLE                      │
│                                                 │
│  [Road markings and boundaries]                │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Use Cases:
- Road maintenance monitoring
- Vehicle suspension adjustment
- Route quality assessment
- Infrastructure reporting

---

### **CAMERA 3+ (Additional Cameras)**
**Focus**: Edge Detection & Night Vision

#### Specialized Features:
- 🌃 **Edge Detection** - Canny edge detection
  - 50/150 threshold
  - Blended with original (70/30)
  - White edge overlay

- 🔦 **Night Vision Mode** - Enhanced low-light
  - Contrast enhancement
  - Edge emphasis

- 🎯 **Object Detection** - Basic detection overlay
  - White bounding boxes
  - Compatible with edge visualization

#### Visual Overlays:
```
┌─────────────────────────────────────────────────┐
│ CAM 3: EDGE DETECTION                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Edge-Enhanced View]                          │
│     ┌──────┐                                   │
│     │      │  (edges highlighted in white)     │
│     └──────┘                                   │
│                                                 │
│  [Detected Objects with White Boxes]           │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Use Cases:
- Low-light operation
- Structural feature detection
- Night driving assistance
- Backup/auxiliary views

---

## 🎮 Additional Visualization Views

These appear when enabled via checkboxes:

### **OPTICAL FLOW VIEW (Toggle)**
Full-screen optical flow visualization
- HSV color-coded flow field
- Magnitude as brightness
- Direction as hue
- Label: "OPTICAL FLOW" (cyan)

### **DANGER HEATMAP VIEW (Toggle)**
Temporal danger zone visualization
- Red intensity = danger level
- Gaussian blobs at collision zones
- Decay over time (95% per frame)
- Label: "DANGER HEATMAP" (red)

---

## 📊 Complete Feature Matrix

| Camera | Object Detection | Tracking | Lanes | Pose | Plates | Flow | Potholes | Edges |
|--------|-----------------|----------|-------|------|--------|------|----------|-------|
| **Primary** | ✅ Full | ✅ Full | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Cam 0** | ✅ Basic | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Cam 1** | ✅ Basic | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Cam 2** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Cam 3+** | ✅ Basic | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Flow View** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Heatmap** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## 🚀 Running the Multi-Camera System

### Launch Command:
```bash
cd /home/vision2030/Desktop/adas-perception
source venv/bin/activate
python3 adas_complete_ultra.py
```

### Camera Selection:
1. Dialog will show available cameras
2. Select 1-4 cameras
3. Choose resolution (640x480 or 1280x720)
4. Click OK

### Enable Features:
- Check "Scene Classification" ✅ (enabled by default)
- Check "Vehicle Types" ✅ (enabled by default)
- Check "Motion Prediction" ✅ (enabled by default)
- Check "Behavior Score" ✅ (enabled by default)
- Check "Optical Flow" ☐ (optional - adds flow view)
- Check "Danger Heatmap" ☐ (optional - adds heatmap view)

### Start Processing:
- Click **▶ START** button
- All cameras begin processing with their specialized features
- Toggle features on/off in real-time

---

## 🎯 Optimal Camera Placement

### Recommended Physical Setup:

```
                    ┌─────────┐
                    │  CAM 0  │  Pedestrian View
                    │ (Front) │  - Crosswalks
                    └─────────┘  - Sidewalks
                         ▲
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐     ┌────┴────┐    ┌────┴────┐
    │  CAM 1  │     │ PRIMARY │    │  CAM 2  │
    │ (Left)  │     │ (Front) │    │ (Right) │
    │ Motion  │     │   Full  │    │  Road   │
    └─────────┘     │Perception│   │ Surface │
                    └─────────┘    └─────────┘
```

#### Camera 0 (Front-High):
- Angle: Slightly downward (10-15°)
- Purpose: Pedestrian detection at crosswalks
- Coverage: Wide field of view

#### PRIMARY (Front-Center):
- Angle: Straight ahead
- Purpose: Main driving perception
- Coverage: Road and traffic

#### Camera 1 (Left/Right):
- Angle: 30-45° to side
- Purpose: Side traffic, lane changes
- Coverage: Adjacent lanes

#### Camera 2 (Front-Low):
- Angle: Downward (30-40°)
- Purpose: Road surface inspection
- Coverage: Immediate road surface

---

## 🎨 Color Coding Reference

### Primary Camera:
- **Green boxes** = Safe objects
- **Orange boxes** = Warning (close)
- **Red boxes** = Critical (collision risk)
- **Yellow lines** = Lane markings
- **Orange dots** = Predicted positions
- **Cyan text** = Scene info
- **Green/Yellow/Red** = Behavior score

### Camera 0 (Pedestrian):
- **Green boxes** = Person/Vehicle detection
- **Cyan dots** = Pose keypoints
- **Yellow boxes** = License plates
- **Orange label** = View identifier

### Camera 1 (Motion):
- **Cyan arrows** = Motion vectors
- **Cyan boxes** = Detected objects
- **Cyan text** = Motion statistics

### Camera 2 (Road):
- **Purple circles** = Potholes
- **Yellow lines** = Lane markings
- **Magenta text** = View identifier

### Camera 3+ (Edge):
- **White boxes** = Object detection
- **White lines** = Edge detection
- **White text** = View identifier

### Additional Views:
- **Cyan label** = Optical Flow
- **Red label** = Danger Heatmap

---

## 💡 Performance Tips

### For Best Multi-Camera Performance:

1. **Resolution**: Use 640x480 for 4 cameras, 1280x720 for 1-2 cameras
2. **Disable Heavy Features**: Turn off pose estimation if FPS drops
3. **Camera Priority**: Primary camera gets most processing power
4. **GPU Acceleration**: Enable CUDA if available
5. **Thread Allocation**: Each camera processes in parallel

### Expected FPS:

| Cameras | Resolution | Features | Expected FPS |
|---------|-----------|----------|--------------|
| 1 | 1280x720 | All | 20-25 |
| 2 | 1280x720 | Most | 15-20 |
| 4 | 640x480 | Selected | 12-18 |
| 4 | 1280x720 | All | 8-12 |

---

## 🔧 Customization

### Modify Camera Assignments:

Edit [adas_complete_ultra.py:338-505](adas_complete_ultra.py#L338-L505) to change which camera has which features:

```python
def _process_secondary_camera(self, frame, idx, cam_id, settings, primary_detections):
    # idx == 0: First secondary camera
    # idx == 1: Second secondary camera
    # idx == 2: Third secondary camera
    # Customize features for each idx
```

### Add New Features:

1. Import feature from ultra modules
2. Initialize in `__init__`
3. Add to appropriate camera in `_process_secondary_camera`
4. Add toggle checkbox in `_add_ultra_controls`

---

## 📈 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UltraMainFrame                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera Manager ─→ Get All Frames                              │
│         ↓                                                       │
│  Primary Frame ──→ Full Perception Pipeline                    │
│         │              ├─ Detection (YOLOv8)                   │
│         │              ├─ Tracking (Kalman)                    │
│         │              ├─ Lanes (Hough)                        │
│         │              ├─ Scene Classification                 │
│         │              ├─ Vehicle Classification               │
│         │              ├─ Motion Prediction                    │
│         │              └─ Behavior Analysis                    │
│         │                                                       │
│  Secondary Frames ──→ Specialized Processing                   │
│         │                                                       │
│         ├─ Camera 0 ──→ Pose + Plates                         │
│         ├─ Camera 1 ──→ Optical Flow                          │
│         ├─ Camera 2 ──→ Potholes + Lanes                      │
│         └─ Camera 3+ ─→ Edge Detection                        │
│                                                                 │
│  Additional Views ──→ Optional Visualizations                  │
│         ├─ Optical Flow (if enabled)                           │
│         └─ Danger Heatmap (if enabled)                         │
│                                                                 │
│  GUI Update ──→ Display All Frames                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Next Steps

1. ✅ Run the application
2. ✅ Select your cameras
3. ✅ Start processing
4. ✅ Watch specialized features on each camera
5. ✅ Toggle features in real-time
6. ✅ Record sessions with all features

**Enjoy your complete multi-camera ADAS system!** 🚀
