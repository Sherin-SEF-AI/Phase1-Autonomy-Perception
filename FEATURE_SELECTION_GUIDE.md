# 🎛️ Feature Selection Guide

## Overview

The ADAS Complete Ultra application now includes a **pre-launch feature selection dialog** that allows you to manually enable/disable features before starting the application. This gives you complete control over which features to use and helps optimize performance.

---

## How to Use

### 1. Launch the Application

```bash
cd /home/vision2030/Desktop/adas-perception
source venv/bin/activate
python3 adas_complete_ultra.py
```

### 2. Feature Selection Dialog Appears

Before the main application window opens, you'll see a **Feature Selection Dialog** with:

- **Title**: "Select Features to Enable"
- **Description**: Explains that more features = lower FPS
- **Feature Categories**: Organized by type with color coding
- **Quick Select Buttons**: Preset configurations
- **FPS Estimate**: Real-time performance prediction
- **Profile Management**: Save/load your configurations

---

## Feature Categories

### 🟠 ULTRA FEATURES (Orange)
Core advanced features from the ultra features module:

- ☑️ **Optical Flow Visualization** - Dense motion field display (Medium: -5 FPS)
- ☑️ **Scene Classification** - Road/highway/urban detection (Low: -2 FPS)
- ☑️ **Vehicle Type Classification** - Car/truck/bus/motorcycle (Medium: -3 FPS)
- ☑️ **Motion Prediction Paths** - Future trajectory visualization (Low: -1 FPS)
- ☑️ **Driving Behavior Score** - Aggressive/normal/cautious (Low: -1 FPS)
- ☑️ **Object Density Heatmap** - Heat map of object locations (Medium: -2 FPS)

### 🟢 AI FEATURES (Green)
Modern AI-powered detection and monitoring:

- ☑️ **Traffic Sign Recognition** - Stop, yield, speed limits (Medium: -4 FPS)
- ☑️ **Traffic Light Detection** - Red/yellow/green state (Medium: -3 FPS)
- ☑️ **Driver Attention Monitoring** - Drowsiness, distraction (High: -6 FPS)
- ☑️ **Weather Condition Detection** - Clear/rainy/foggy/snowy (Very low: -0.5 FPS)
- ☑️ **Parking Space Detection** - Available/occupied spaces (High: -5 FPS)
- ☑️ **Night Vision Enhancement** - Low-light image enhancement (Low: -1 FPS)
- ☑️ **Emergency Vehicle Detection** - Flashing lights detection (Low: -2 FPS)
- ☑️ **Road Debris Detection** - Obstacles on road (Medium: -4 FPS)

### 🔵 MULTI-CAMERA FEATURES (Blue)
Specialized processing for additional cameras:

- ☐ **Camera 0: Pedestrian Pose** - 33-point body keypoints (High: -7 FPS)
- ☐ **Camera 0: License Plates** - Vehicle plate detection (Low: -2 FPS)
- ☐ **Camera 1: Optical Flow** - Dense motion on secondary camera (Medium: -5 FPS)
- ☐ **Camera 2: Pothole Detection** - Road surface damage (Medium: -3 FPS)

### 🟣 VISUALIZATION (Magenta)
Dashboard and analytics displays:

- ☑️ **Statistics Dashboard** - Real-time metrics panel (Very low impact)
- ☑️ **Performance Graphs** - FPS and CPU graphs over time (Very low impact)

---

## Quick Select Presets

Use these buttons for instant configuration:

### 🚀 **Speed Mode**
**Target: 25-30 FPS**

Enables only lightweight features:
- Scene Classification ✅
- Driving Behavior Score ✅
- Weather Detection ✅
- Night Vision ✅

**Best for:**
- Real-time demos
- Lower-powered hardware
- Smooth video playback
- Testing basic functionality

---

### ⚖️ **Balanced** (Recommended)
**Target: 18-22 FPS**

Enables all features EXCEPT heavy ones:
- ❌ Optical Flow Visualization
- ❌ Object Density Heatmap
- ❌ Driver Attention Monitoring
- ❌ Parking Space Detection
- ❌ All Multi-Camera Features

**Best for:**
- General use
- Good feature set with acceptable performance
- Most users should start here
- 1-2 cameras

---

### 🎯 **All Features**
**Target: 8-15 FPS**

Enables EVERYTHING - all 100+ features

**Best for:**
- Feature demonstrations
- Research and development
- Powerful hardware (i9, 32GB RAM, GPU)
- Can accept slower FPS
- Maximum capabilities showcase

---

### ⚪ **Disable All**
**Target: 30-40 FPS**

Disables all ultra/AI features, keeps only core ADAS:
- Object detection
- Tracking
- Lane detection
- Collision warning

**Best for:**
- Troubleshooting
- Baseline performance testing
- Fastest possible operation

---

## FPS Estimate Display

The dialog shows a **real-time FPS estimate** that updates as you check/uncheck features:

### Color Coding:
- 🟢 **Green** (≥25 FPS): "Excellent" - Smooth, real-time performance
- 🟡 **Yellow** (20-24 FPS): "Good" - Very usable, minor lag
- 🟠 **Orange** (15-19 FPS): "Acceptable" - Noticeable lag, still usable
- 🔴 **Red** (<15 FPS): "Slow" - Consider disabling features

### How It Works:
The estimate uses a performance model based on:
- Base FPS: 30 (with core features only)
- Each feature has a measured FPS impact
- Total impact is summed and subtracted from base
- Minimum shown is 8 FPS (worst case)

**Note**: Actual FPS depends on your hardware, camera count, and resolution. The estimate is a guideline.

---

## Profile Management

### Save Your Configuration

1. Check the features you want
2. Click **"Save Profile"**
3. Choose a filename (e.g., `my_config.profile`)
4. Click Save

Your configuration is saved as a JSON file.

### Load a Saved Configuration

1. Click **"Load Profile"**
2. Select a `.profile` file
3. Click Open

All checkboxes update to match the saved configuration.

### Profile File Format

Profiles are simple JSON files:

```json
{
  "show_optical_flow": false,
  "show_scene_info": true,
  "classify_vehicles": true,
  "traffic_signs": true,
  "traffic_lights": true,
  "weather_detection": true,
  ...
}
```

You can edit these manually if needed!

---

## Example Profiles

### Profile 1: "Demo Mode"
**For smooth presentations**

```
✅ Scene Classification
✅ Vehicle Classification
✅ Motion Predictions
✅ Behavior Score
✅ Traffic Signs
✅ Traffic Lights
✅ Weather Detection
✅ Night Vision
✅ Statistics Dashboard
✅ Performance Graphs
```

**Expected FPS**: 20-25

---

### Profile 2: "Research Mode"
**For data collection and analysis**

```
✅ All AI Features
✅ All Visualization Features
✅ Scene Classification
✅ Vehicle Classification
✅ Motion Predictions
✅ Behavior Score
❌ Heavy processing (optical flow, heatmap)
❌ Multi-camera features (unless needed)
```

**Expected FPS**: 15-20

---

### Profile 3: "Night Driving"
**Optimized for low-light conditions**

```
✅ Night Vision Enhancement (essential!)
✅ Scene Classification
✅ Traffic Signs
✅ Traffic Lights
✅ Weather Detection
✅ Emergency Vehicles
✅ Debris Detection
❌ Driver Monitoring (hard in dark)
❌ Parking Spaces (hard in dark)
```

**Expected FPS**: 20-25

---

### Profile 4: "Parking Lot"
**Focused on parking assistance**

```
✅ Parking Space Detection
✅ Scene Classification
✅ Emergency Vehicles
✅ Traffic Signs
❌ Motion Prediction (not needed when slow)
❌ Behavior Score (not needed when slow)
❌ Traffic Lights (usually not in lots)
```

**Expected FPS**: 18-22

---

## Starting the Application

### After Selecting Features:

1. Review your selections
2. Check the FPS estimate
3. Click **"Start Application"** (or press Enter)

The main ADAS window will open with your selected features enabled.

### To Cancel:

Click **"Cancel"** or press Escape - the application will exit.

---

## Runtime Control

Even after selecting features at startup, you can still **toggle features on/off during runtime** using the checkboxes in the main application's control panel:

- **ULTRA FEATURES** section (orange labels)
- **AI FEATURES** section (green labels)

However, **pre-selecting features at launch** is recommended because:
- ✅ Better performance (doesn't initialize unused features)
- ✅ Cleaner interface (only shows enabled features)
- ✅ Easier to manage with profiles
- ✅ FPS estimate helps you plan

---

## Performance Tips

### For Maximum Speed:
1. Use **"Speed Mode"** preset
2. Use **1 camera only**
3. Choose **640x480 resolution**
4. Disable multi-camera features

**Expected**: 25-30 FPS

### For Maximum Features:
1. Use **"All Features"** preset
2. Use **1-2 cameras** (not 4)
3. Use **640x480 resolution** (not 1280x720)
4. Have powerful hardware

**Expected**: 12-18 FPS

### For Balanced Experience:
1. Use **"Balanced"** preset (default)
2. Use **1-2 cameras**
3. Use **1280x720 resolution**
4. Enable features as needed

**Expected**: 18-22 FPS

---

## Troubleshooting

### Dialog Doesn't Appear

**Problem**: Application launches directly without showing feature selection

**Solution**: The dialog should appear automatically. If not:
- Check that you're running `adas_complete_ultra.py` (not `adas-perception.py` or `adas_fast.py`)
- Check console for error messages

### FPS Lower Than Expected

**Problem**: Application runs slower than the estimate

**Possible Causes**:
1. **Camera resolution too high** - Try 640x480 instead of 1280x720
2. **Too many cameras** - Estimate assumes 1-2 cameras
3. **Background applications** - Close other programs
4. **Weak CPU** - See hardware recommendations in `WHICH_VERSION_TO_RUN.txt`

**Solutions**:
- Disable more features
- Use fewer cameras
- Lower camera resolution
- Try `adas_fast.py` instead

### Profile Won't Load

**Problem**: Error when loading a saved profile

**Solutions**:
- Ensure the `.profile` file is valid JSON
- Check that feature names in the file match current features
- Re-save the profile if it's from an older version

---

## Feature Impact Reference

Quick reference for FPS impact:

| Impact Level | FPS Loss | Examples |
|--------------|----------|----------|
| **Very Low** | -0.5 to -1 FPS | Weather, Night Vision, Stats |
| **Low** | -1 to -2 FPS | Predictions, Behavior, Emergency, Plates |
| **Medium** | -3 to -5 FPS | Signs, Lights, Vehicles, Flow, Debris, Pothole |
| **High** | -6 to -7 FPS | Driver Monitoring, Parking, Pedestrian Pose |

---

## Recommended Workflows

### First Time Users:
1. Launch application
2. Select **"Balanced"** preset
3. Click **"Start Application"**
4. Observe performance
5. Adjust if needed and save as profile

### Regular Users:
1. Launch application
2. Click **"Load Profile"**
3. Select your saved configuration
4. Click **"Start Application"**

### Testing New Features:
1. Launch application
2. Select **"Balanced"** or **"Speed Mode"**
3. Enable ONE additional feature to test
4. Observe FPS impact
5. Keep or disable based on results

---

## Summary

The Feature Selection Dialog gives you:

✅ **Control** - Choose exactly which features to enable
✅ **Performance** - See FPS estimates before starting
✅ **Convenience** - Save/load configurations as profiles
✅ **Flexibility** - Quick presets or custom selection
✅ **Transparency** - Clear impact information for each feature

**Recommended for most users**: Start with **"Balanced"** preset, then customize as needed!

---

## Related Documentation

- [WHICH_VERSION_TO_RUN.txt](WHICH_VERSION_TO_RUN.txt) - Choosing the right ADAS version
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - Performance tuning guide
- [AI_FEATURES_GUIDE.md](AI_FEATURES_GUIDE.md) - Detailed AI feature documentation
- [MULTI_CAMERA_GUIDE.md](MULTI_CAMERA_GUIDE.md) - Multi-camera setup guide

---

**Happy driving! 🚗**
