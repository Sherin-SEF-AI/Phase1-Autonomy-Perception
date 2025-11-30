#!/bin/bash
#
# ADAS Perception System - Launch Script
# Version: 2.0.0
#

echo "=========================================="
echo "  ADAS Perception System v2.0"
echo "  Enterprise-Grade Autonomous Perception"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python version: $PYTHON_VERSION"

# Check for required modules
echo ""
echo "Checking dependencies..."

check_module() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(python3 -c "import $1; print($1.__version__ if hasattr($1, '__version__') else 'installed')" 2>/dev/null)
        echo "  ✓ $1: $VERSION"
        return 0
    else
        echo "  ✗ $1: NOT INSTALLED"
        return 1
    fi
}

# Check core dependencies
MISSING=0
check_module "numpy" || MISSING=1
check_module "cv2" || MISSING=1
check_module "wx" || MISSING=1
check_module "psutil" || MISSING=1

# Check optional dependencies
echo ""
echo "Optional dependencies:"
check_module "ultralytics"
check_module "torch"

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Some required dependencies are missing"
    echo "Install them with: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting ADAS Perception System..."
echo "=========================================="
echo ""

# Set environment variables
export OPENCV_VIDEOIO_PRIORITY_V4L2=1
export OPENCV_LOG_LEVEL=ERROR

# Run the application
python3 adas-perception.py "$@"

# Exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Application exited with code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
