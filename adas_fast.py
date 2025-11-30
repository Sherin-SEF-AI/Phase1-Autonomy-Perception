#!/usr/bin/env python3
"""
ADAS FAST - Performance-Optimized Version
Targets 25-30 FPS with essential features only

Run: python3 adas_fast.py
"""

import wx
import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

print("=" * 70)
print("  ADAS FAST - Performance Optimized")
print("=" * 70)

# Import base application
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_app", "adas-perception.py")
    base_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_app)
    print("✅ Base application loaded")
except Exception as e:
    print(f"❌ Could not load base app: {e}")
    sys.exit(1)

print("=" * 70)


class FastMainFrame(base_app.MainFrame):
    """Optimized main frame for maximum performance"""

    def __init__(self):
        # Initialize base class
        super().__init__()

        # Set title
        self.SetTitle("ADAS FAST - Performance Optimized (30+ FPS)")

        # PERFORMANCE SETTINGS - All heavy features disabled
        self.fast_mode = True
        self.frame_skip = 1  # Process every frame (no skipping for basic features)
        self.frame_count = 0

        # Override settings for performance
        self.settings.update({
            "enable_detection": True,   # Keep
            "enable_tracking": True,    # Keep
            "enable_lanes": True,       # Keep
            "confidence_threshold": 0.5,  # Higher = fewer detections = faster
            "ego_speed": 60,
        })

        print("\n" + "=" * 70)
        print("  FAST MODE SETTINGS:")
        print("=" * 70)
        print("  ✅ Object Detection (YOLOv8)")
        print("  ✅ Multi-Object Tracking")
        print("  ✅ Lane Detection")
        print("  ✅ Collision Warning")
        print("  ✅ Distance Estimation")
        print("  ❌ All advanced features DISABLED for speed")
        print("=" * 70)
        print("\n  Target: 25-30 FPS")
        print("  Recommended: 1 camera @ 640x480 or 1280x720")
        print("=" * 70)
        print("\n🚀 Application ready! Click START to begin.\n")

    def _processing_loop(self):
        """Optimized processing loop - essential features only"""
        while self.running:
            try:
                frames = self.camera_manager.get_all_frames()

                if not frames:
                    time.sleep(0.001)  # Minimal sleep
                    continue

                self.frame_count += 1

                primary_frame = frames.get(self.primary_camera)
                if primary_frame is not None:
                    settings = self.settings.copy()

                    # Update perception engine
                    self.perception_engine.detector.confidence_threshold = settings["confidence_threshold"]
                    self.perception_engine.collision_system.set_ego_speed(settings["ego_speed"])

                    # CORE PROCESSING ONLY - No extras
                    result_frame, metrics = self.perception_engine.process_frame(
                        primary_frame,
                        enable_detection=settings["enable_detection"],
                        enable_lanes=settings["enable_lanes"],
                        enable_tracking=settings["enable_tracking"]
                    )

                    # NO ULTRA FEATURES - Maximum performance

                    # Record if active
                    if self.video_recorder.is_recording:
                        self.video_recorder.write_frame(result_frame)

                    # Create secondary frames dict (minimal processing)
                    secondary_frames = {}
                    camera_ids = sorted([k for k in frames.keys() if k != self.primary_camera])

                    # Show other cameras RAW (no processing)
                    for cam_id in camera_ids[:3]:  # Max 3 secondary cameras
                        if cam_id in frames:
                            secondary_frames[cam_id] = frames[cam_id]

                    # Post update event
                    evt = base_app.FrameUpdateEvent(
                        primary_frame=result_frame,
                        secondary_frames=secondary_frames,
                        bev_frame=None
                    )
                    wx.PostEvent(self, evt)

                    # Post metrics
                    metrics_evt = base_app.MetricsUpdateEvent(metrics=metrics)
                    wx.PostEvent(self, metrics_evt)

            except Exception as e:
                import logging
                logging.getLogger('ADAS').error(f"Processing error: {e}")
                time.sleep(0.01)


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  Starting ADAS FAST Application")
    print("=" * 70)

    # Create application
    app = wx.App()

    # Set dark theme
    if hasattr(wx, 'SystemOptions'):
        wx.SystemOptions.SetOption("msw.dark-mode", 2)

    # Create and show frame
    frame = FastMainFrame()
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
