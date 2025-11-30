#!/usr/bin/env python3
"""
Quick launcher for ADAS Complete Ultra with ALL features enabled by default
Skips the feature selection dialog
"""

import wx
import sys

# Import the main application
import adas_complete_ultra

# All features enabled configuration
ALL_FEATURES = {
    # Ultra features
    'show_optical_flow': True,
    'show_scene_info': True,
    'classify_vehicles': True,
    'show_predictions': True,
    'show_behavior_score': True,
    'show_heatmap': True,
    # AI features
    'traffic_signs': True,
    'traffic_lights': True,
    'driver_monitoring': True,
    'weather_detection': True,
    'parking_spaces': True,
    'night_vision': True,
    'emergency_vehicles': True,
    'debris_detection': True,
    # Multi-camera features
    'camera_pedestrian_pose': True,
    'camera_license_plates': True,
    'camera_optical_flow': True,
    'camera_pothole': True,
    # Visualization
    'show_stats_dashboard': True,
    'show_performance_graphs': True,
}

def main():
    """Launch with all features enabled"""
    print("\n" + "=" * 70)
    print("  ADAS COMPLETE ULTRA - ALL FEATURES MODE")
    print("=" * 70)
    print("  Launching with ALL 100+ features enabled...")
    print("  Expected FPS: 8-15 (depending on hardware)")
    print("=" * 70)

    # Create application
    app = wx.App()

    # Set dark theme
    if hasattr(wx, 'SystemOptions'):
        wx.SystemOptions.SetOption("msw.dark-mode", 2)

    # Create frame with all features enabled
    frame = adas_complete_ultra.UltraMainFrame(selected_features=ALL_FEATURES)
    frame.Show()
    frame.Center()

    # Start main loop
    app.MainLoop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutdown requested...exiting")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
