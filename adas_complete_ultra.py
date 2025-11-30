#!/usr/bin/env python3
"""
COMPLETE ULTRA-ADVANCED ADAS APPLICATION
Full integration with ALL advanced features visible in GUI

Run: python3 adas_complete_ultra.py
"""

import wx
import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

print("=" * 70)
print("  ADAS COMPLETE ULTRA - Loading...")
print("=" * 70)

# Import base application
try:
    # We'll extend the working v1.0 app
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_app", "adas-perception.py")
    base_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_app)
    print("✅ Base application loaded")
except Exception as e:
    print(f"❌ Could not load base app: {e}")
    print("Please ensure adas-perception.py is in the same directory")
    sys.exit(1)

# Import ultra features
try:
    from ultra_features import (
        OpticalFlowAnalyzer,
        VehicleTypeClassifier,
        MotionPredictor,
        DrivingBehaviorAnalyzer,
        LicensePlateDetector,
        PotholeDetector,
        SuddenMovementDetector
    )
    print("✅ Ultra features loaded")
    HAS_ULTRA_FEATURES = True
except Exception as e:
    print(f"⚠️  Ultra features not available: {e}")
    HAS_ULTRA_FEATURES = False

try:
    from adas_ultra_advanced import (
        SceneClassifier,
        PedestrianPoseEstimator,
        PointCloudReconstructor,
        PanoramaStitcher
    )
    print("✅ Ultra advanced modules loaded")
    HAS_ULTRA_ADVANCED = True
except Exception as e:
    print(f"⚠️  Ultra advanced not available: {e}")
    HAS_ULTRA_ADVANCED = False

try:
    from ultra_visualization import (
        HeatmapTimeline,
        StatisticsDashboard
    )
    print("✅ Ultra visualization loaded")
    HAS_ULTRA_VIZ = True
except Exception as e:
    print(f"⚠️  Ultra visualization not available: {e}")
    HAS_ULTRA_VIZ = False

print("=" * 70)


# ============================================================================
# FEATURE SELECTION DIALOG
# ============================================================================

class FeatureSelectionDialog(wx.Dialog):
    """Pre-launch dialog for selecting features to enable"""

    def __init__(self, parent=None):
        super().__init__(parent, title="ADAS Feature Selection",
                        size=(700, 800),
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.selected_features = {}
        self._create_ui()
        self.Centre()

    def _create_ui(self):
        """Create the feature selection interface"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(panel, label="Select Features to Enable")
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)

        # Description
        desc = wx.StaticText(panel, label="Choose which features to enable. More features = lower FPS")
        desc.SetForegroundColour(wx.Colour(150, 150, 150))
        main_sizer.Add(desc, 0, wx.ALL | wx.CENTER, 5)

        # Scrolled window for features
        scroll = wx.ScrolledWindow(panel)
        scroll.SetScrollRate(5, 5)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        # Quick selection buttons
        quick_sizer = wx.BoxSizer(wx.HORIZONTAL)
        quick_label = wx.StaticText(scroll, label="Quick Select:")
        quick_sizer.Add(quick_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        btn_all = wx.Button(scroll, label="All Features", size=(120, 30))
        btn_balanced = wx.Button(scroll, label="Balanced", size=(120, 30))
        btn_speed = wx.Button(scroll, label="Speed Mode", size=(120, 30))
        btn_none = wx.Button(scroll, label="Disable All", size=(120, 30))

        quick_sizer.Add(btn_all, 0, wx.ALL, 5)
        quick_sizer.Add(btn_balanced, 0, wx.ALL, 5)
        quick_sizer.Add(btn_speed, 0, wx.ALL, 5)
        quick_sizer.Add(btn_none, 0, wx.ALL, 5)

        scroll_sizer.Add(quick_sizer, 0, wx.ALL | wx.EXPAND, 10)
        scroll_sizer.Add(wx.StaticLine(scroll), 0, wx.ALL | wx.EXPAND, 5)

        # Feature categories
        self.checkboxes = {}

        # ULTRA FEATURES
        self._add_category(scroll_sizer, scroll, "ULTRA FEATURES", wx.Colour(255, 165, 0), [
            ('show_optical_flow', 'Optical Flow Visualization', 'Medium impact (-5 FPS)'),
            ('show_scene_info', 'Scene Classification', 'Low impact (-2 FPS)'),
            ('classify_vehicles', 'Vehicle Type Classification', 'Medium impact (-3 FPS)'),
            ('show_predictions', 'Motion Prediction Paths', 'Low impact (-1 FPS)'),
            ('show_behavior_score', 'Driving Behavior Score', 'Low impact (-1 FPS)'),
            ('show_heatmap', 'Object Density Heatmap', 'Medium impact (-2 FPS)'),
        ], True)

        # AI FEATURES
        self._add_category(scroll_sizer, scroll, "AI FEATURES", wx.Colour(0, 255, 128), [
            ('traffic_signs', 'Traffic Sign Recognition', 'Medium impact (-4 FPS)'),
            ('traffic_lights', 'Traffic Light Detection', 'Medium impact (-3 FPS)'),
            ('driver_monitoring', 'Driver Attention Monitoring', 'High impact (-6 FPS)'),
            ('weather_detection', 'Weather Condition Detection', 'Very low impact (-0.5 FPS)'),
            ('parking_spaces', 'Parking Space Detection', 'High impact (-5 FPS)'),
            ('night_vision', 'Night Vision Enhancement', 'Low impact (-1 FPS)'),
            ('emergency_vehicles', 'Emergency Vehicle Detection', 'Low impact (-2 FPS)'),
            ('debris_detection', 'Road Debris Detection', 'Medium impact (-4 FPS)'),
        ], True)

        # MULTI-CAMERA FEATURES
        self._add_category(scroll_sizer, scroll, "MULTI-CAMERA FEATURES", wx.Colour(100, 200, 255), [
            ('camera_pedestrian_pose', 'Camera 0: Pedestrian Pose', 'High impact (-7 FPS)'),
            ('camera_license_plates', 'Camera 0: License Plates', 'Low impact (-2 FPS)'),
            ('camera_optical_flow', 'Camera 1: Optical Flow', 'Medium impact (-5 FPS)'),
            ('camera_pothole', 'Camera 2: Pothole Detection', 'Medium impact (-3 FPS)'),
        ], False)

        # VISUALIZATION FEATURES
        self._add_category(scroll_sizer, scroll, "VISUALIZATION", wx.Colour(255, 100, 255), [
            ('show_stats_dashboard', 'Statistics Dashboard', 'Very low impact'),
            ('show_performance_graphs', 'Performance Graphs', 'Very low impact'),
        ], True)

        scroll.SetSizer(scroll_sizer)
        main_sizer.Add(scroll, 1, wx.ALL | wx.EXPAND, 10)

        # Expected FPS display
        self.fps_label = wx.StaticText(panel, label="Expected FPS: Calculating...")
        self.fps_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(self.fps_label, 0, wx.ALL | wx.CENTER, 10)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_load = wx.Button(panel, label="Load Profile", size=(120, 35))
        btn_save = wx.Button(panel, label="Save Profile", size=(120, 35))
        btn_sizer.Add(btn_load, 0, wx.ALL, 5)
        btn_sizer.Add(btn_save, 0, wx.ALL, 5)
        btn_sizer.AddStretchSpacer()

        btn_cancel = wx.Button(panel, wx.ID_CANCEL, "Cancel", size=(120, 35))
        btn_ok = wx.Button(panel, wx.ID_OK, "Start Application", size=(150, 35))
        btn_ok.SetDefault()

        btn_sizer.Add(btn_cancel, 0, wx.ALL, 5)
        btn_sizer.Add(btn_ok, 0, wx.ALL, 5)

        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(main_sizer)

        # Event bindings
        btn_all.Bind(wx.EVT_BUTTON, lambda e: self._select_preset('all'))
        btn_balanced.Bind(wx.EVT_BUTTON, lambda e: self._select_preset('balanced'))
        btn_speed.Bind(wx.EVT_BUTTON, lambda e: self._select_preset('speed'))
        btn_none.Bind(wx.EVT_BUTTON, lambda e: self._select_preset('none'))
        btn_load.Bind(wx.EVT_BUTTON, self._load_profile)
        btn_save.Bind(wx.EVT_BUTTON, self._save_profile)

        # Update FPS when checkboxes change
        for cb in self.checkboxes.values():
            cb.Bind(wx.EVT_CHECKBOX, lambda e: self._update_fps_estimate())

        # Set default preset
        self._select_preset('balanced')

    def _add_category(self, sizer, parent, title, color, features, default_enabled):
        """Add a feature category section"""
        # Category title
        cat_title = wx.StaticText(parent, label=title)
        cat_title.SetForegroundColour(color)
        cat_font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        cat_title.SetFont(cat_font)
        sizer.Add(cat_title, 0, wx.ALL, 5)

        # Feature checkboxes
        for key, label, impact in features:
            box_sizer = wx.BoxSizer(wx.HORIZONTAL)

            cb = wx.CheckBox(parent, label=label)
            cb.SetValue(default_enabled)
            self.checkboxes[key] = cb

            box_sizer.Add(cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            box_sizer.AddStretchSpacer()

            impact_label = wx.StaticText(parent, label=impact)
            impact_label.SetForegroundColour(wx.Colour(150, 150, 150))
            impact_label.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
            box_sizer.Add(impact_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

            sizer.Add(box_sizer, 0, wx.ALL | wx.EXPAND, 2)

        sizer.Add(wx.StaticLine(parent), 0, wx.ALL | wx.EXPAND, 10)

    def _select_preset(self, preset):
        """Apply a feature preset"""
        if preset == 'all':
            # Enable everything
            for cb in self.checkboxes.values():
                cb.SetValue(True)
        elif preset == 'none':
            # Disable everything
            for cb in self.checkboxes.values():
                cb.SetValue(False)
        elif preset == 'speed':
            # Speed mode - only essential features
            speed_features = ['show_scene_info', 'show_behavior_score',
                            'weather_detection', 'night_vision']
            for key, cb in self.checkboxes.items():
                cb.SetValue(key in speed_features)
        elif preset == 'balanced':
            # Balanced - good features without heavy ones
            heavy_features = ['show_optical_flow', 'show_heatmap', 'driver_monitoring',
                            'parking_spaces', 'camera_pedestrian_pose', 'camera_optical_flow',
                            'camera_pothole']
            for key, cb in self.checkboxes.items():
                cb.SetValue(key not in heavy_features)

        self._update_fps_estimate()

    def _update_fps_estimate(self):
        """Update estimated FPS based on selected features"""
        # Base FPS with core features only
        base_fps = 30

        # FPS impact of each feature
        fps_impact = {
            'show_optical_flow': 5,
            'show_scene_info': 2,
            'classify_vehicles': 3,
            'show_predictions': 1,
            'show_behavior_score': 1,
            'show_heatmap': 2,
            'traffic_signs': 4,
            'traffic_lights': 3,
            'driver_monitoring': 6,
            'weather_detection': 0.5,
            'parking_spaces': 5,
            'night_vision': 1,
            'emergency_vehicles': 2,
            'debris_detection': 4,
            'camera_pedestrian_pose': 7,
            'camera_license_plates': 2,
            'camera_optical_flow': 5,
            'camera_pothole': 3,
            'show_stats_dashboard': 0.5,
            'show_performance_graphs': 0.5,
        }

        # Calculate total FPS loss
        total_loss = 0
        for key, cb in self.checkboxes.items():
            if cb.GetValue() and key in fps_impact:
                total_loss += fps_impact[key]

        estimated_fps = max(8, base_fps - total_loss)

        # Update label with color coding
        if estimated_fps >= 25:
            color = wx.Colour(0, 200, 0)  # Green
            status = "Excellent"
        elif estimated_fps >= 20:
            color = wx.Colour(200, 200, 0)  # Yellow
            status = "Good"
        elif estimated_fps >= 15:
            color = wx.Colour(255, 150, 0)  # Orange
            status = "Acceptable"
        else:
            color = wx.Colour(255, 0, 0)  # Red
            status = "Slow"

        self.fps_label.SetLabel(f"Expected FPS: {estimated_fps:.0f} ({status})")
        self.fps_label.SetForegroundColour(color)

    def _load_profile(self, event):
        """Load feature profile from file"""
        with wx.FileDialog(self, "Load Feature Profile",
                          wildcard="Profile files (*.profile)|*.profile",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'r') as f:
                    import json
                    profile = json.load(f)
                    for key, value in profile.items():
                        if key in self.checkboxes:
                            self.checkboxes[key].SetValue(value)
                    self._update_fps_estimate()
                    wx.MessageBox(f"Profile loaded from {pathname}", "Success", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"Error loading profile: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _save_profile(self, event):
        """Save feature profile to file"""
        with wx.FileDialog(self, "Save Feature Profile",
                          wildcard="Profile files (*.profile)|*.profile",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            try:
                import json
                profile = {key: cb.GetValue() for key, cb in self.checkboxes.items()}
                with open(pathname, 'w') as f:
                    json.dump(profile, f, indent=2)
                wx.MessageBox(f"Profile saved to {pathname}", "Success", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"Error saving profile: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def get_selected_features(self):
        """Return dictionary of selected features"""
        return {key: cb.GetValue() for key, cb in self.checkboxes.items()}


# ============================================================================
# ENHANCED MAIN FRAME WITH ULTRA FEATURES
# ============================================================================

class UltraMainFrame(base_app.MainFrame):
    """Enhanced main frame with all ultra features"""

    def __init__(self, selected_features=None):
        # Initialize base class
        super().__init__()

        # Set new title
        self.SetTitle("ADAS COMPLETE ULTRA - All Advanced Features")

        # Ultra feature flags (define BEFORE adding controls)
        # Use selected features from dialog if provided, otherwise use defaults
        default_settings = {
            # Original features
            'show_optical_flow': False,
            'show_scene_info': True,
            'classify_vehicles': True,
            'show_predictions': True,
            'show_behavior_score': True,
            'show_heatmap': False,
            # AI features
            'traffic_signs': True,
            'traffic_lights': True,
            'driver_monitoring': False,  # Optional - for driver camera
            'weather_detection': True,
            'parking_spaces': False,
            'night_vision': True,
            'emergency_vehicles': True,
            'debris_detection': True,
            # Multi-camera features
            'camera_pedestrian_pose': False,
            'camera_license_plates': False,
            'camera_optical_flow': False,
            'camera_pothole': False,
            # Visualization
            'show_stats_dashboard': True,
            'show_performance_graphs': True,
        }

        # If selected_features provided from dialog, use those
        if selected_features:
            self.ultra_settings = selected_features
        else:
            self.ultra_settings = default_settings

        # Initialize ultra components
        self.optical_flow = OpticalFlowAnalyzer() if HAS_ULTRA_FEATURES else None
        self.scene_classifier = SceneClassifier() if HAS_ULTRA_ADVANCED else None
        self.vehicle_classifier = VehicleTypeClassifier() if HAS_ULTRA_FEATURES else None
        self.motion_predictor = MotionPredictor() if HAS_ULTRA_FEATURES else None
        self.behavior_analyzer = DrivingBehaviorAnalyzer() if HAS_ULTRA_FEATURES else None
        self.heatmap = HeatmapTimeline() if HAS_ULTRA_VIZ else None

        # Initialize AI features
        try:
            from ultra_ai_features import (
                TrafficSignRecognizer,
                TrafficLightDetector,
                DriverAttentionMonitor,
                WeatherDetector,
                ParkingSpaceDetector,
                EmergencyVehicleDetector,
                NightVisionEnhancer,
                DebrisDetector
            )
            self.traffic_sign_detector = TrafficSignRecognizer()
            self.traffic_light_detector = TrafficLightDetector()
            self.driver_monitor = DriverAttentionMonitor()
            self.weather_detector = WeatherDetector()
            self.parking_detector = ParkingSpaceDetector()
            self.emergency_detector = EmergencyVehicleDetector()
            self.night_enhancer = NightVisionEnhancer()
            self.debris_detector = DebrisDetector()
            print("  ✅ AI Features initialized")
        except Exception as e:
            print(f"  ⚠️  AI Features not available: {e}")
            self.traffic_sign_detector = None
            self.traffic_light_detector = None
            self.driver_monitor = None
            self.weather_detector = None
            self.parking_detector = None
            self.emergency_detector = None
            self.night_enhancer = None
            self.debris_detector = None

        # Add ultra feature toggles to control panel
        self._add_ultra_controls()

        print("\n" + "=" * 70)
        print("  ULTRA FEATURES ENABLED:")
        print("=" * 70)
        if self.optical_flow:
            print("  ✅ Optical Flow Visualization")
        if self.scene_classifier:
            print("  ✅ Scene Classification (Time/Road/Weather/Traffic)")
        if self.vehicle_classifier:
            print("  ✅ Vehicle Type Classification")
        if self.motion_predictor:
            print("  ✅ Motion Prediction & Trajectories")
        if self.behavior_analyzer:
            print("  ✅ Driving Behavior Analysis")
        if self.heatmap:
            print("  ✅ Danger Zone Heatmap")
        print("=" * 70)
        print("\n🚀 Application ready! Click START to begin.\n")

    def _add_ultra_controls(self):
        """Add ultra feature controls to the control panel"""
        # Find the control panel sizer
        if hasattr(self, 'control_panel'):
            sizer = self.control_panel.GetSizer()

            # Add separator
            line = wx.StaticLine(self.control_panel)
            sizer.Add(line, 0, wx.ALL | wx.EXPAND, 10)

            # Ultra features title
            ultra_title = wx.StaticText(self.control_panel, label="ULTRA FEATURES")
            ultra_title.SetForegroundColour(wx.Colour(255, 165, 0))
            ultra_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            sizer.Add(ultra_title, 0, wx.ALL | wx.ALIGN_CENTER, 5)

            # Ultra feature checkboxes
            ultra_features = [
                ('show_optical_flow', 'Optical Flow'),
                ('show_scene_info', 'Scene Classification'),
                ('classify_vehicles', 'Vehicle Types'),
                ('show_predictions', 'Motion Prediction'),
                ('show_behavior_score', 'Behavior Score'),
                ('show_heatmap', 'Danger Heatmap'),
            ]

            for key, label in ultra_features:
                cb = wx.CheckBox(self.control_panel, label=label)
                cb.SetValue(self.ultra_settings.get(key, False))
                cb.SetForegroundColour(wx.Colour(200, 200, 200))
                cb.Bind(wx.EVT_CHECKBOX, lambda e, k=key: self._on_ultra_toggle(k, e))
                sizer.Add(cb, 0, wx.ALL | wx.EXPAND, 5)

            # AI Features separator
            line2 = wx.StaticLine(self.control_panel)
            sizer.Add(line2, 0, wx.ALL | wx.EXPAND, 10)

            # AI features title
            ai_title = wx.StaticText(self.control_panel, label="AI FEATURES")
            ai_title.SetForegroundColour(wx.Colour(0, 255, 128))
            ai_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            sizer.Add(ai_title, 0, wx.ALL | wx.ALIGN_CENTER, 5)

            # AI feature checkboxes
            ai_features = [
                ('traffic_signs', 'Traffic Signs'),
                ('traffic_lights', 'Traffic Lights'),
                ('weather_detection', 'Weather Detection'),
                ('night_vision', 'Night Vision'),
                ('emergency_vehicles', 'Emergency Vehicles'),
                ('debris_detection', 'Debris Detection'),
                ('parking_spaces', 'Parking Spaces'),
                ('driver_monitoring', 'Driver Monitoring'),
            ]

            for key, label in ai_features:
                cb = wx.CheckBox(self.control_panel, label=label)
                cb.SetValue(self.ultra_settings.get(key, False))
                cb.SetForegroundColour(wx.Colour(150, 255, 150))
                cb.Bind(wx.EVT_CHECKBOX, lambda e, k=key: self._on_ultra_toggle(k, e))
                sizer.Add(cb, 0, wx.ALL | wx.EXPAND, 5)

            self.control_panel.Layout()

    def _on_ultra_toggle(self, key: str, event):
        """Handle ultra feature toggle"""
        self.ultra_settings[key] = event.IsChecked()
        logger = __import__('logging').getLogger('ADAS')
        logger.info(f"Ultra feature {key}: {event.IsChecked()}")

    def _processing_loop(self):
        """Enhanced processing loop with ultra features for ALL cameras"""
        while self.running:
            try:
                frames = self.camera_manager.get_all_frames()

                if not frames:
                    time.sleep(0.01)
                    continue

                primary_frame = frames.get(self.primary_camera)
                if primary_frame is not None:
                    settings = self.settings.copy()

                    # Update perception engine
                    self.perception_engine.detector.confidence_threshold = settings["confidence_threshold"]
                    self.perception_engine.collision_system.set_ego_speed(settings["ego_speed"])

                    # Process PRIMARY camera with full perception
                    result_frame, metrics = self.perception_engine.process_frame(
                        primary_frame,
                        enable_detection=settings["enable_detection"],
                        enable_lanes=settings["enable_lanes"],
                        enable_tracking=settings["enable_tracking"]
                    )

                    # === PRIMARY CAMERA ULTRA FEATURES ===
                    detections = self.perception_engine.detector.detect(primary_frame) if settings["enable_detection"] else []
                    tracked_objects = list(self.perception_engine.tracker.objects.values()) if settings["enable_tracking"] else []

                    # 1. Scene Classification
                    if self.scene_classifier and self.ultra_settings['show_scene_info']:
                        scene_context = self.scene_classifier.classify(primary_frame, detections)
                        self._draw_scene_info(result_frame, scene_context)

                    # 2. Vehicle Classification
                    if self.vehicle_classifier and self.ultra_settings['classify_vehicles']:
                        for det in detections:
                            if hasattr(det, 'class_name') and 'car' in det.class_name.lower():
                                v_type = self.vehicle_classifier.classify(det.bbox, primary_frame)
                                x1, y1, x2, y2 = det.bbox
                                cv2.putText(result_frame, v_type, (x1, y2 + 15),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

                    # 3. Motion Prediction
                    if self.motion_predictor and self.ultra_settings['show_predictions']:
                        for obj in tracked_objects:
                            predictions = self.motion_predictor.predict(obj, num_steps=5)
                            for i, (px, py) in enumerate(predictions):
                                if 0 <= px < result_frame.shape[1] and 0 <= py < result_frame.shape[0]:
                                    cv2.circle(result_frame, (px, py), 2, (255, 165, 0), -1)

                    # 4. Driving Behavior
                    if self.behavior_analyzer and self.ultra_settings['show_behavior_score']:
                        behavior = self.behavior_analyzer.analyze(
                            tracked_objects,
                            self.perception_engine.lane_detector if hasattr(self.perception_engine, 'lane_detector') else None,
                            1 if metrics.collision_risk != "NONE" else 0
                        )
                        score = self.behavior_analyzer.get_score()
                        self._draw_behavior_score(result_frame, behavior, score)

                    # === AI FEATURES START ===

                    # Apply night vision enhancement first if enabled
                    if self.night_enhancer and self.ultra_settings['night_vision']:
                        result_frame = self.night_enhancer.enhance(result_frame)

                    # 5. Traffic Sign Recognition
                    if self.traffic_sign_detector and self.ultra_settings['traffic_signs']:
                        traffic_signs = self.traffic_sign_detector.detect_signs(primary_frame)
                        self._draw_traffic_signs(result_frame, traffic_signs)

                    # 6. Traffic Light Detection
                    if self.traffic_light_detector and self.ultra_settings['traffic_lights']:
                        traffic_lights = self.traffic_light_detector.detect_lights(primary_frame)
                        self._draw_traffic_lights(result_frame, traffic_lights)

                    # 7. Weather Detection
                    current_weather = None
                    if self.weather_detector and self.ultra_settings['weather_detection']:
                        current_weather = self.weather_detector.detect_weather(primary_frame)
                        self._draw_weather_info(result_frame, current_weather)

                    # 8. Emergency Vehicle Detection
                    if self.emergency_detector and self.ultra_settings['emergency_vehicles']:
                        emergency_vehicles = self.emergency_detector.detect_emergency_vehicles(
                            primary_frame, detections)
                        self._draw_emergency_vehicles(result_frame, emergency_vehicles)

                    # 9. Debris Detection
                    if self.debris_detector and self.ultra_settings['debris_detection']:
                        debris_list = self.debris_detector.detect_debris(primary_frame)
                        self._draw_debris(result_frame, debris_list)

                    # 10. Parking Space Detection (if enabled)
                    if self.parking_detector and self.ultra_settings['parking_spaces']:
                        parking_spaces = self.parking_detector.detect_spaces(primary_frame)
                        self._draw_parking_spaces(result_frame, parking_spaces)

                    # === AI FEATURES END ===

                    # === PROCESS SECONDARY CAMERAS WITH SPECIALIZED FEATURES ===
                    secondary_frames = {}
                    camera_ids = sorted([k for k in frames.keys() if k != self.primary_camera])

                    for idx, cam_id in enumerate(camera_ids):
                        frame = frames[cam_id]
                        processed = self._process_secondary_camera(frame, idx, cam_id, settings, detections)
                        secondary_frames[cam_id] = processed

                    # === GENERATE ADDITIONAL VISUALIZATION VIEWS ===

                    # Optical Flow View (if enabled and space available)
                    if self.optical_flow and self.ultra_settings['show_optical_flow']:
                        flow_data = self.optical_flow.calculate(primary_frame)
                        if flow_data and len(secondary_frames) < 3:
                            flow_vis = self.optical_flow.visualize(flow_data, primary_frame.shape)
                            self._add_view_label(flow_vis, "OPTICAL FLOW", (0, 255, 255))
                            secondary_frames[999] = flow_vis

                    # Heatmap View (if enabled and space available)
                    if self.heatmap and self.ultra_settings['show_heatmap']:
                        collision_zones = [(obj.centroid[0], obj.centroid[1], 50)
                                         for obj in tracked_objects if obj.ttc < 3.0]
                        self.heatmap.update(collision_zones, primary_frame.shape)

                        if len(secondary_frames) < 3:
                            heatmap_vis = self.heatmap.visualize()
                            if heatmap_vis is not None:
                                self._add_view_label(heatmap_vis, "DANGER HEATMAP", (0, 0, 255))
                                secondary_frames[998] = heatmap_vis

                    # Record if active
                    if self.video_recorder.is_recording:
                        self.video_recorder.write_frame(result_frame)

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
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _draw_scene_info(self, frame, scene_context):
        """Draw scene classification info on frame"""
        y_offset = 60
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Background panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (350, 50 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw info
        info_items = [
            f"Time: {scene_context.time_of_day.name}",
            f"Road: {scene_context.road_type.name}",
            f"Traffic: {scene_context.traffic_density.name}",
            f"Condition: {scene_context.road_condition.name}",
            f"Visibility: {scene_context.visibility_score:.2f}"
        ]

        for i, text in enumerate(info_items):
            cv2.putText(frame, text, (20, y_offset + i * 22),
                       font, font_scale, (0, 255, 255), thickness)

    def _draw_behavior_score(self, frame, behavior, score):
        """Draw driving behavior score"""
        # Position in top-right
        x = frame.shape[1] - 200
        y = 60

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 30), (x + 190, y + 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Score color
        if score < 60:
            color = (0, 0, 255)  # Red
        elif score < 85:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green

        # Draw score
        cv2.putText(frame, f"Behavior: {behavior}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}/100", (x, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _process_secondary_camera(self, frame: np.ndarray, idx: int, cam_id: int,
                                   settings: dict, primary_detections: list) -> np.ndarray:
        """
        Process secondary camera with specialized features based on camera position

        Camera Assignment:
        - Camera 0 (idx=0): Pedestrian Pose Estimation + License Plate Detection
        - Camera 1 (idx=1): Optical Flow Analysis + Motion Vectors
        - Camera 2 (idx=2): Pothole Detection + Road Surface Analysis
        - Additional: Night Vision / Edge Detection
        """
        processed_frame = frame.copy()

        try:
            # === CAMERA 0: PEDESTRIAN & LICENSE PLATE FOCUS ===
            if idx == 0:
                self._add_view_label(processed_frame, f"CAM {cam_id}: PEDESTRIAN + PLATES", (255, 165, 0))

                # Detect objects in this camera
                if settings["enable_detection"]:
                    detections = self.perception_engine.detector.detect(frame)

                    # Draw basic detections
                    for det in detections:
                        if hasattr(det, 'bbox') and hasattr(det, 'class_name'):
                            x1, y1, x2, y2 = det.bbox

                            # Pedestrian pose estimation
                            if 'person' in det.class_name.lower() and HAS_ULTRA_ADVANCED:
                                try:
                                    from adas_ultra_advanced import PedestrianPoseEstimator
                                    if not hasattr(self, 'pose_estimator'):
                                        self.pose_estimator = PedestrianPoseEstimator()

                                    pose = self.pose_estimator.estimate(frame, det.bbox)
                                    if pose and pose.keypoints:
                                        # Draw skeleton
                                        for kp in pose.keypoints:
                                            if kp[2] > 0.5:  # Confidence threshold
                                                cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

                                        # Draw action label
                                        cv2.putText(processed_frame, pose.action, (x1, y1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                except:
                                    pass

                            # License plate detection for vehicles
                            if 'car' in det.class_name.lower() and HAS_ULTRA_FEATURES:
                                try:
                                    from ultra_features import LicensePlateDetector
                                    if not hasattr(self, 'plate_detector'):
                                        self.plate_detector = LicensePlateDetector()

                                    plate_bbox = self.plate_detector.detect(frame, det.bbox)
                                    if plate_bbox:
                                        px1, py1, px2, py2 = plate_bbox
                                        cv2.rectangle(processed_frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                                        cv2.putText(processed_frame, "PLATE", (px1, py1 - 5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                except:
                                    pass

                            # Draw bounding box
                            color = (0, 255, 0)
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(processed_frame, det.class_name, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # === CAMERA 1: OPTICAL FLOW & MOTION ANALYSIS ===
            elif idx == 1:
                self._add_view_label(processed_frame, f"CAM {cam_id}: MOTION ANALYSIS", (0, 255, 255))

                # Optical flow visualization
                if self.optical_flow and HAS_ULTRA_FEATURES:
                    try:
                        flow_data = self.optical_flow.calculate(frame)
                        if flow_data:
                            # Draw motion vectors
                            flow = flow_data['flow']
                            step = 16
                            for y in range(0, processed_frame.shape[0], step):
                                for x in range(0, processed_frame.shape[1], step):
                                    if y < flow.shape[0] and x < flow.shape[1]:
                                        fx, fy = flow[y, x]
                                        magnitude = np.sqrt(fx**2 + fy**2)
                                        if magnitude > 1.0:
                                            end_x = int(x + fx * 2)
                                            end_y = int(y + fy * 2)
                                            cv2.arrowedLine(processed_frame, (x, y), (end_x, end_y),
                                                          (0, 255, 255), 1, tipLength=0.3)

                            # Display motion stats
                            cv2.putText(processed_frame, f"Motion: {flow_data['dominant_motion']}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(processed_frame, f"Avg: {flow_data['avg_magnitude']:.1f} px/frame", (10, 55),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    except:
                        pass

                # Basic detection with motion emphasis
                if settings["enable_detection"]:
                    detections = self.perception_engine.detector.detect(frame)
                    for det in detections:
                        if hasattr(det, 'bbox'):
                            x1, y1, x2, y2 = det.bbox
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # === CAMERA 2: ROAD SURFACE & POTHOLE DETECTION ===
            elif idx == 2:
                self._add_view_label(processed_frame, f"CAM {cam_id}: ROAD SURFACE", (255, 0, 255))

                # Pothole detection
                if HAS_ULTRA_FEATURES:
                    try:
                        from ultra_features import PotholeDetector
                        if not hasattr(self, 'pothole_detector'):
                            self.pothole_detector = PotholeDetector()

                        potholes = self.pothole_detector.detect(frame)
                        for (x, y, radius) in potholes:
                            cv2.circle(processed_frame, (x, y), radius, (255, 0, 255), 2)
                            cv2.putText(processed_frame, "POTHOLE", (x - 30, y - radius - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                        # Display count
                        cv2.putText(processed_frame, f"Potholes: {len(potholes)}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    except:
                        pass

                # Lane detection
                if settings["enable_lanes"]:
                    try:
                        lanes = self.perception_engine.lane_detector.detect_lanes(frame)
                        if lanes is not None and len(lanes) > 0:
                            for lane in lanes:
                                if len(lane) > 0:
                                    pts = np.array(lane, np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(processed_frame, [pts], False, (255, 255, 0), 2)
                    except:
                        pass

            # === ADDITIONAL CAMERAS: EDGE DETECTION / NIGHT VISION ===
            else:
                self._add_view_label(processed_frame, f"CAM {cam_id}: EDGE DETECTION", (255, 255, 255))

                # Edge detection overlay
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                # Blend with original
                processed_frame = cv2.addWeighted(processed_frame, 0.7, edges_colored, 0.3, 0)

                # Basic detection
                if settings["enable_detection"]:
                    detections = self.perception_engine.detector.detect(frame)
                    for det in detections:
                        if hasattr(det, 'bbox'):
                            x1, y1, x2, y2 = det.bbox
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        except Exception as e:
            import logging
            logging.getLogger('ADAS').error(f"Error processing camera {cam_id}: {e}")

        return processed_frame

    def _add_view_label(self, frame: np.ndarray, label: str, color: tuple):
        """Add a label to identify the view type"""
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text
        cv2.putText(frame, label, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ========================================================================
    # AI FEATURE DRAWING METHODS
    # ========================================================================

    def _draw_traffic_signs(self, frame: np.ndarray, signs: List):
        """Draw detected traffic signs"""
        for sign in signs:
            x1, y1, x2, y2 = sign.bbox

            # Color based on sign type
            from ultra_ai_features import TrafficSignType
            if sign.sign_type == TrafficSignType.STOP:
                color = (0, 0, 255)  # Red
                label = "STOP"
            elif sign.sign_type == TrafficSignType.SPEED_LIMIT:
                color = (0, 165, 255)  # Orange
                label = f"{sign.speed_limit} km/h"
            elif sign.sign_type == TrafficSignType.YIELD:
                color = (0, 255, 255)  # Yellow
                label = "YIELD"
            else:
                color = (255, 255, 255)  # White
                label = sign.sign_type.value

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _draw_traffic_lights(self, frame: np.ndarray, lights: List):
        """Draw detected traffic lights"""
        for light in lights:
            x1, y1, x2, y2 = light.bbox

            # Color based on state
            from ultra_ai_features import TrafficLightState
            if light.state == TrafficLightState.RED:
                color = (0, 0, 255)
                label = "RED"
            elif light.state == TrafficLightState.YELLOW:
                color = (0, 255, 255)
                label = "YELLOW"
            elif light.state == TrafficLightState.GREEN:
                color = (0, 255, 0)
                label = "GREEN"
            else:
                color = (128, 128, 128)
                label = "UNKNOWN"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw state label
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_weather_info(self, frame: np.ndarray, weather):
        """Draw weather information"""
        if weather is None:
            return

        # Position in top-right, below behavior score
        x = frame.shape[1] - 200
        y = 130

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 20), (x + 190, y + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Weather icon/text
        weather_text = f"Weather: {weather.value}"

        # Color based on weather
        if weather.value == "CLEAR":
            color = (0, 255, 255)  # Yellow
        elif weather.value == "RAINY":
            color = (255, 100, 0)  # Blue
        elif weather.value == "FOGGY":
            color = (200, 200, 200)  # Gray
        else:
            color = (255, 255, 255)  # White

        cv2.putText(frame, weather_text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_emergency_vehicles(self, frame: np.ndarray, vehicles: List):
        """Draw emergency vehicle alerts"""
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']

            # Flashing red alert
            color = (0, 0, 255)  # Red

            # Draw thick box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

            # Alert text
            cv2.putText(frame, "EMERGENCY!", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        # If any emergency vehicles detected, show alert
        if vehicles:
            alert_text = f"! {len(vehicles)} EMERGENCY VEHICLE(S) !"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            x = (frame.shape[1] - text_size[0]) // 2
            y = 100

            # Flashing background
            cv2.rectangle(frame, (x - 10, y - 40), (x + text_size[0] + 10, y + 10), (0, 0, 255), -1)
            cv2.putText(frame, alert_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    def _draw_debris(self, frame: np.ndarray, debris_list: List):
        """Draw road debris warnings"""
        for debris_bbox in debris_list:
            x1, y1, x2, y2 = debris_bbox

            # Orange warning
            color = (0, 165, 255)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Warning icon/text
            cv2.putText(frame, "DEBRIS!", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_parking_spaces(self, frame: np.ndarray, spaces: List):
        """Draw detected parking spaces"""
        for space in spaces:
            # Draw corners
            corners = np.array(space.corners, np.int32)
            corners = corners.reshape((-1, 1, 2))

            # Color based on occupancy
            if space.is_occupied:
                color = (0, 0, 255)  # Red - occupied
                label = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green - available
                label = "AVAILABLE"

            # Draw space outline
            cv2.polylines(frame, [corners], True, color, 2)

            # Draw label
            if space.corners:
                x, y = space.corners[0]
                cv2.putText(frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_driver_attention(self, frame: np.ndarray, attention):
        """Draw driver attention monitoring info"""
        if attention is None:
            return

        # Position in bottom-left
        x = 20
        y = frame.shape[0] - 100

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 60), (x + 250, y + 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Alert color based on level
        if attention.alert_level == 0:
            color = (0, 255, 0)  # Green
        elif attention.alert_level == 1:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Draw state
        state_text = f"Driver: {attention.state.value}"
        cv2.putText(frame, state_text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw eye closure
        cv2.putText(frame, f"Eyes: {int((1-attention.eye_closure)*100)}%", (x, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Alert if needed
        if attention.alert_level >= 2:
            alert_text = "ATTENTION REQUIRED!"
            cv2.putText(frame, alert_text, (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  Starting ADAS Complete Ultra Application")
    print("=" * 70)

    # Create application
    app = wx.App()

    # Set dark theme
    if hasattr(wx, 'SystemOptions'):
        wx.SystemOptions.SetOption("msw.dark-mode", 2)

    # Show feature selection dialog
    print("\n  Opening feature selection dialog...")
    dialog = FeatureSelectionDialog()

    if dialog.ShowModal() == wx.ID_OK:
        # User confirmed - get selected features
        selected_features = dialog.get_selected_features()
        dialog.Destroy()

        print("\n" + "=" * 70)
        print("  Feature Selection Summary:")
        print("=" * 70)
        enabled_count = sum(1 for v in selected_features.values() if v)
        print(f"  {enabled_count} features enabled")
        print("=" * 70)

        # Create and show main frame with selected features
        frame = UltraMainFrame(selected_features=selected_features)
        frame.Show()
        frame.Center()

        # Start main loop
        app.MainLoop()
    else:
        # User cancelled
        dialog.Destroy()
        print("\n  Application launch cancelled by user")
        print("=" * 70)


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
