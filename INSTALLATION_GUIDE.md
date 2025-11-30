# ADAS Perception System - Installation Guide

## Complete Setup Instructions for All Platforms

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Linux Installation](#linux-installation)
3. [Windows Installation](#windows-installation)
4. [macOS Installation](#macos-installation)
5. [Docker Installation](#docker-installation)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum System
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **Processor**: Intel i5 or AMD Ryzen 5 (quad-core)
- **RAM**: 8 GB
- **GPU**: Integrated graphics (for CPU mode)
- **Storage**: 5 GB free space
- **Camera**: Any USB webcam (640x480 @ 15fps)
- **Python**: 3.8 or higher

### Recommended System
- **OS**: Ubuntu 22.04 LTS
- **Processor**: Intel i7-10700K or AMD Ryzen 7 3700X
- **RAM**: 16 GB
- **GPU**: NVIDIA GTX 1660 or better (6GB VRAM)
- **Storage**: 20 GB free space (SSD recommended)
- **Camera**: 4x USB 3.0 cameras (1080p @ 30fps)
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU acceleration)

---

## Linux Installation

### Ubuntu 20.04 / 22.04

#### Step 1: Update System
```bash
sudo apt update && sudo apt upgrade -y
```

#### Step 2: Install System Dependencies
```bash
# Install Python and build tools
sudo apt install -y python3 python3-pip python3-dev python3-venv

# Install OpenCV dependencies
sudo apt install -y \
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

# Install wxPython dependencies
sudo apt install -y \
    libgtk-3-dev \
    libwebkit2gtk-4.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    freeglut3-dev \
    libnotify-dev \
    libsdl2-dev

# Install additional tools
sudo apt install -y \
    git \
    v4l-utils \
    ffmpeg
```

#### Step 3: Create Virtual Environment
```bash
cd /home/vision2030/Desktop/adas-perception

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Step 4: Install Python Packages
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78
pip install wxPython==4.2.1
pip install psutil==5.9.5

# Install optional dependencies
pip install ultralytics==8.0.200
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install analytics dependencies
pip install matplotlib seaborn plotly pandas scipy scikit-learn
```

#### Step 5: Grant Camera Permissions
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set camera permissions
sudo chmod 666 /dev/video*

# Reload groups (or logout/login)
newgrp video
```

#### Step 6: Test Installation
```bash
# Run test
python3 -c "import cv2, wx, numpy, ultralytics; print('✓ All dependencies OK')"

# Test camera access
v4l2-ctl --list-devices
```

---

## Windows Installation

### Windows 10 / 11

#### Step 1: Install Python
1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Click "Install Now"

#### Step 2: Install Visual C++ Build Tools
1. Download from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install "Desktop development with C++"
3. Restart computer

#### Step 3: Open PowerShell as Administrator
```powershell
# Navigate to project directory
cd C:\Users\YourName\Desktop\adas-perception

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Step 4: Install Python Packages
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install numpy opencv-python opencv-contrib-python
pip install wxPython psutil

# Install optional dependencies
pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install analytics
pip install matplotlib seaborn pandas
```

#### Step 5: Install Camera Drivers
- Install manufacturer's camera drivers
- For generic USB cameras, Windows should auto-install

#### Step 6: Test Installation
```powershell
# Test dependencies
python -c "import cv2, wx, numpy; print('OK')"

# List cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

---

## macOS Installation

### macOS 10.15+ (Catalina, Big Sur, Monterey, Ventura)

#### Step 1: Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Dependencies
```bash
# Install Python
brew install python@3.10

# Install OpenCV dependencies
brew install opencv

# Install additional tools
brew install ffmpeg
```

#### Step 3: Create Virtual Environment
```bash
cd ~/Desktop/adas-perception

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

#### Step 4: Install Python Packages
```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install numpy opencv-python opencv-contrib-python
pip install wxPython psutil

# Install optional
pip install ultralytics
pip install torch torchvision

# Install analytics
pip install matplotlib seaborn pandas
```

#### Step 5: Grant Camera Permissions
1. System Preferences → Security & Privacy
2. Camera tab
3. ✅ Enable for Terminal/Python

#### Step 6: Test Installation
```bash
python3 -c "import cv2, wx, numpy; print('✓ OK')"
```

---

## Docker Installation

### Using Docker (All Platforms)

#### Step 1: Install Docker
- Linux: `sudo apt install docker.io`
- Windows/Mac: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)

#### Step 2: Create Dockerfile
```dockerfile
# Save as: Dockerfile
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libopencv-dev \
    libgtk-3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy opencv-python opencv-contrib-python \
    wxPython psutil ultralytics \
    matplotlib seaborn pandas

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Expose display
ENV DISPLAY=:0

CMD ["python3", "adas-perception.py"]
```

#### Step 3: Build and Run
```bash
# Build image
docker build -t adas-perception .

# Run container (Linux)
docker run -it --rm \
    --device=/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    adas-perception

# Run container (Windows with WSL2)
docker run -it --rm adas-perception
```

---

## Verification

### Check All Components

```bash
cd /home/vision2030/Desktop/adas-perception

# Run verification script
python3 << 'EOF'
import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError:
    print("✗ NumPy NOT INSTALLED")

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError:
    print("✗ OpenCV NOT INSTALLED")

try:
    import wx
    print("✓ wxPython:", wx.__version__)
except ImportError:
    print("✗ wxPython NOT INSTALLED")

try:
    import psutil
    print("✓ psutil:", psutil.__version__)
except ImportError:
    print("✗ psutil NOT INSTALLED")

try:
    import ultralytics
    print("✓ Ultralytics:", ultralytics.__version__)
except ImportError:
    print("⚠ Ultralytics not installed (optional)")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except ImportError:
    print("⚠ PyTorch not installed (optional)")

print("\n--- Camera Check ---")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera {i} detected")
        cap.release()

print("\nVerification complete!")
EOF
```

Expected output:
```
Python version: 3.10.x
✓ NumPy: 1.24.3
✓ OpenCV: 4.8.1
✓ wxPython: 4.2.1
✓ psutil: 5.9.5
✓ Ultralytics: 8.0.200
✓ PyTorch: 2.1.0
  CUDA available: True

--- Camera Check ---
✓ Camera 0 detected

Verification complete!
```

---

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected (Linux)
```bash
# Check camera devices
ls -l /dev/video*

# Test camera
ffplay /dev/video0

# Fix permissions
sudo chmod 666 /dev/video*
sudo usermod -a -G video $USER
```

#### 2. wxPython Installation Failed
```bash
# Install dependencies first
sudo apt install libgtk-3-dev libwebkit2gtk-4.0-dev

# Install from wheel
pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 wxPython
```

#### 3. CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Import Errors
```bash
# Reinstall from requirements
pip install --force-reinstall -r requirements.txt

# Or install individually
pip install --force-reinstall opencv-python
```

#### 5. Low FPS / Performance
- Lower resolution (640x480)
- Disable unnecessary features
- Use GPU acceleration
- Check CPU usage: `top` or Task Manager
- Close other applications

#### 6. Memory Errors
- Reduce number of cameras
- Disable data logging
- Decrease history length:
  ```python
  tracker = ObjectTracker(max_disappeared=10)  # Reduce from 30
  ```

---

## Hardware-Specific Setup

### NVIDIA Jetson Nano
```bash
# Install JetPack
sudo apt install nvidia-jetpack

# Install Python packages
pip3 install numpy opencv-python psutil

# wxPython (may need custom build)
pip3 install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04 wxPython
```

### Raspberry Pi 4
```bash
# Enable camera
sudo raspi-config
# → Interface Options → Camera → Enable

# Install dependencies
sudo apt install python3-opencv python3-numpy

# Reduce requirements
# Use lighter detection model or disable features
```

---

## Post-Installation

### 1. Download YOLO Model
```bash
cd /home/vision2030/Desktop/adas-perception

# YOLOv8n will auto-download on first run, or:
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 2. Test Run
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Run application
python3 adas-perception.py
```

### 3. Create Desktop Shortcut (Linux)
```bash
cat > ~/Desktop/ADAS-Perception.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=ADAS Perception System
Comment=Advanced Driver Assistance System
Exec=/home/vision2030/Desktop/adas-perception/venv/bin/python /home/vision2030/Desktop/adas-perception/adas-perception.py
Icon=video-display
Terminal=false
Categories=Development;Science;
EOF

chmod +x ~/Desktop/ADAS-Perception.desktop
```

---

## Next Steps

✅ Installation complete! Now you can:

1. **Run Basic Example**
   ```bash
   cd /home/vision2030/Desktop/adas-perception
   python3 examples/basic_usage.py
   ```

2. **Run Advanced Analytics**
   ```bash
   python3 examples/advanced_analytics.py
   ```

3. **Launch Full Application**
   ```bash
   ./run.sh
   # or
   python3 adas-perception.py
   ```

4. **Read Documentation**
   - [README_ADVANCED.md](README_ADVANCED.md) - Full feature list
   - [FEATURES_COMPARISON.md](FEATURES_COMPARISON.md) - v1 vs v2

---

## Support

- 📧 **Email**: support@deepmost.ai
- 📚 **Docs**: [Full Documentation](README_ADVANCED.md)
- 🐛 **Issues**: Report bugs in GitHub Issues
- 💬 **Community**: Join our Discord

---

**Installation Guide v2.0**
*Last updated: 2025-01-29*
