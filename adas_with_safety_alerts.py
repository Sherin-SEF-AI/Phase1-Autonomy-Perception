#!/usr/bin/env python3
"""
ADAS with Integrated Safety Alerts
Includes FCW, LDW, and DMS with audio warnings

Run: python3 adas_with_safety_alerts.py
"""

import wx
import cv2
import numpy as np
import sys
from pathlib import Path

# Import base application
import importlib.util
spec = importlib.util.spec_from_file_location("base_app", "adas-perception.py")
base_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_app)

# Import safety alert system
from safety_alerts import SafetyAlertSystem, extract_lane_center, create_driver_state_from_dms

# Import DMS if available
try:
    from ultra_ai_features import DriverAttentionMonitor
    HAS_DMS = True
except:
    HAS_DMS = False
    print("⚠️  Driver Attention Monitor not available")


class SafetyADASFrame(base_app.MainFrame):
    """Enhanced ADAS with integrated safety alerts"""

    def __init__(self):
        super().__init__()

        # Initialize safety alert system
        self.safety_system = SafetyAlertSystem()

        # Initialize DMS if available
        if HAS_DMS:
            self.dms = DriverAttentionMonitor()
        else:
            self.dms = None

        # Update title
        self.SetTitle("ADAS with Safety Alerts - FCW | LDW | DMS")

        # Add safety controls to control panel
        self._add_safety_controls()

        # Current alert (for display)
        self.current_alert = None
        self.alert_display_time = 0

        print("\n" + "=" * 70)
        print("  SAFETY FEATURES ENABLED")
        print("=" * 70)
        print("  ✓ Forward Collision Warning (FCW)")
        print("  ✓ Lane Departure Warning (LDW)")
        if HAS_DMS:
            print("  ✓ Driver Monitoring System (DMS)")
        print("=" * 70)

    def _add_safety_controls(self):
        """Add safety feature controls to GUI"""
        # Get the control panel sizer
        sizer = self.control_panel.GetSizer()

        # Add separator
        line = wx.StaticLine(self.control_panel)
        sizer.Add(line, 0, wx.ALL | wx.EXPAND, 10)

        # Safety features title
        safety_title = wx.StaticText(self.control_panel, label="SAFETY ALERTS")
        safety_title.SetForegroundColour(wx.Colour(255, 0, 0))  # Red
        safety_font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        safety_title.SetFont(safety_font)
        sizer.Add(safety_title, 0, wx.ALL, 5)

        # FCW checkbox
        self.fcw_checkbox = wx.CheckBox(self.control_panel, label="Forward Collision Warning (FCW)")
        self.fcw_checkbox.SetValue(True)
        self.fcw_checkbox.Bind(wx.EVT_CHECKBOX, self._on_fcw_toggle)
        sizer.Add(self.fcw_checkbox, 0, wx.ALL, 5)

        # LDW checkbox
        self.ldw_checkbox = wx.CheckBox(self.control_panel, label="Lane Departure Warning (LDW)")
        self.ldw_checkbox.SetValue(True)
        self.ldw_checkbox.Bind(wx.EVT_CHECKBOX, self._on_ldw_toggle)
        sizer.Add(self.ldw_checkbox, 0, wx.ALL, 5)

        # DMS checkbox
        self.dms_checkbox = wx.CheckBox(self.control_panel, label="Driver Monitoring System (DMS)")
        self.dms_checkbox.SetValue(HAS_DMS)
        self.dms_checkbox.Enable(HAS_DMS)
        self.dms_checkbox.Bind(wx.EVT_CHECKBOX, self._on_dms_toggle)
        sizer.Add(self.dms_checkbox, 0, wx.ALL, 5)

        # Audio alerts checkbox
        self.audio_checkbox = wx.CheckBox(self.control_panel, label="Audio Alerts")
        self.audio_checkbox.SetValue(True)
        self.audio_checkbox.Bind(wx.EVT_CHECKBOX, self._on_audio_toggle)
        sizer.Add(self.audio_checkbox, 0, wx.ALL, 5)

        # Turn signal simulator (for testing)
        turn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        turn_label = wx.StaticText(self.control_panel, label="Turn Signal:")
        self.turn_left_btn = wx.Button(self.control_panel, label="◄ Left", size=(80, 25))
        self.turn_right_btn = wx.Button(self.control_panel, label="Right ►", size=(80, 25))
        self.turn_off_btn = wx.Button(self.control_panel, label="Off", size=(60, 25))

        self.turn_left_btn.Bind(wx.EVT_BUTTON, lambda e: self._set_turn_signal(True, 'left'))
        self.turn_right_btn.Bind(wx.EVT_BUTTON, lambda e: self._set_turn_signal(True, 'right'))
        self.turn_off_btn.Bind(wx.EVT_BUTTON, lambda e: self._set_turn_signal(False))

        turn_sizer.Add(turn_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        turn_sizer.Add(self.turn_left_btn, 0, wx.ALL, 2)
        turn_sizer.Add(self.turn_right_btn, 0, wx.ALL, 2)
        turn_sizer.Add(self.turn_off_btn, 0, wx.ALL, 2)
        sizer.Add(turn_sizer, 0, wx.ALL, 5)

        self.control_panel.Layout()

    def _on_fcw_toggle(self, event):
        """Toggle FCW"""
        self.safety_system.enable_fcw(self.fcw_checkbox.GetValue())

    def _on_ldw_toggle(self, event):
        """Toggle LDW"""
        self.safety_system.enable_ldw(self.ldw_checkbox.GetValue())

    def _on_dms_toggle(self, event):
        """Toggle DMS"""
        self.safety_system.enable_dms(self.dms_checkbox.GetValue())

    def _on_audio_toggle(self, event):
        """Toggle audio alerts"""
        self.safety_system.enable_audio(self.audio_checkbox.GetValue())

    def _set_turn_signal(self, active, direction=None):
        """Set turn signal state"""
        self.safety_system.set_turn_signal(active, direction)

        # Update button colors
        if active:
            if direction == 'left':
                self.turn_left_btn.SetBackgroundColour(wx.Colour(0, 255, 0))
                self.turn_right_btn.SetBackgroundColour(wx.NullColour)
            else:
                self.turn_right_btn.SetBackgroundColour(wx.Colour(0, 255, 0))
                self.turn_left_btn.SetBackgroundColour(wx.NullColour)
        else:
            self.turn_left_btn.SetBackgroundColour(wx.NullColour)
            self.turn_right_btn.SetBackgroundColour(wx.NullColour)

        self.turn_left_btn.Refresh()
        self.turn_right_btn.Refresh()

    def _process_frame(self, frame, camera_index=0):
        """Override to add safety checks"""
        # Call base processing
        frame = super()._process_frame(frame, camera_index)

        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Only process primary camera for safety alerts
        if camera_index == 0:
            # Check Forward Collision Warning
            if hasattr(self, 'current_detections'):
                fcw_alert = self.safety_system.check_forward_collision(
                    self.current_detections, w, h
                )
                if fcw_alert:
                    self.current_alert = fcw_alert
                    self.alert_display_time = 0

            # Check Lane Departure Warning
            if hasattr(self, 'lane_lines') and self.lane_lines:
                lane_center = extract_lane_center(self.lane_lines)
                frame_center = w / 2

                ldw_alert = self.safety_system.check_lane_departure(
                    lane_center, frame_center, w
                )
                if ldw_alert:
                    self.current_alert = ldw_alert
                    self.alert_display_time = 0

            # Check Driver Monitoring (if DMS available)
            if self.dms and HAS_DMS:
                # Run DMS on frame
                dms_result = self.dms.analyze_driver(frame)
                driver_state = create_driver_state_from_dms(dms_result)

                dms_alert = self.safety_system.check_driver_attention(driver_state)
                if dms_alert:
                    self.current_alert = dms_alert
                    self.alert_display_time = 0

            # Draw alert overlay if active
            if self.current_alert:
                frame = self.safety_system.draw_alert(frame, self.current_alert)
                self.alert_display_time += 1

                # Clear alert after 90 frames (~3 seconds)
                if self.alert_display_time > 90:
                    self.current_alert = None

            # Draw safety system status
            status_text = self.safety_system.get_status_text()
            cv2.putText(frame, status_text, (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  ADAS with Integrated Safety Alerts")
    print("=" * 70)

    # Create application
    app = wx.App()

    # Set dark theme
    if hasattr(wx, 'SystemOptions'):
        wx.SystemOptions.SetOption("msw.dark-mode", 2)

    # Create and show frame
    frame = SafetyADASFrame()
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
