#!/usr/bin/env python3
"""
Enterprise-Grade Advanced Autonomous Vehicle Perception System
Level 2-3 Autonomy - Complete Perception & Planning Stack

Advanced Features:
- Multi-camera management with calibration tools
- Real-time object detection with YOLOv8
- Semantic segmentation for scene understanding
- Monocular depth estimation
- Multi-object tracking with Kalman filtering
- Sensor fusion (Camera + Radar/LiDAR simulation)
- Lane detection with polynomial curve fitting
- Collision warning with TTC and FCW
- Bird's eye view transformation
- Traffic sign recognition & classification
- Distance & speed estimation
- Path prediction & trajectory planning
- Occupancy grid mapping
- 3D bounding boxes
- Heat maps & attention maps
- Data logging & playback system
- Performance profiler
- Comprehensive analytics dashboard
- Video recording with metadata

Author: DeepMost AI Perception Team
Version: 2.0.0
License: MIT
"""

import wx
import wx.lib.newevent
import wx.lib.agw.speedmeter as SM
import wx.lib.plot as plot
import cv2
import numpy as np
import threading
import queue
import time
import json
import os
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Deque
from collections import deque, OrderedDict
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
import hashlib
import gzip

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adas_perception.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ADAS_Perception')

# Custom events for thread-safe GUI updates
FrameUpdateEvent, EVT_FRAME_UPDATE = wx.lib.newevent.NewEvent()
MetricsUpdateEvent, EVT_METRICS_UPDATE = wx.lib.newevent.NewEvent()
AlertEvent, EVT_ALERT = wx.lib.newevent.NewEvent()
AnalyticsUpdateEvent, EVT_ANALYTICS_UPDATE = wx.lib.newevent.NewEvent()


# ============================================================================
# DATA CLASSES & ENUMS
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = auto()
    WARNING = auto()
    DANGER = auto()
    CRITICAL = auto()


class ObjectClass(Enum):
    """Detected object classes relevant to driving"""
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9
    STOP_SIGN = 11


class DrivingBehavior(Enum):
    """Driving behavior classification"""
    NORMAL = auto()
    AGGRESSIVE = auto()
    CAUTIOUS = auto()
    EMERGENCY = auto()


@dataclass
class Detection:
    """Single object detection with 3D information"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    distance: float = 0.0
    velocity: float = 0.0
    track_id: int = -1
    ttc: float = float('inf')
    bbox_3d: Optional[np.ndarray] = None  # 3D bounding box corners
    depth: float = 0.0
    lateral_velocity: float = 0.0
    heading: float = 0.0


@dataclass
class LaneInfo:
    """Lane detection results with advanced features"""
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    center_lane: Optional[np.ndarray] = None
    center_offset: float = 0.0
    curvature: float = 0.0
    lane_width: float = 3.7
    confidence: float = 0.0
    departure_warning: bool = False
    left_departure: bool = False
    right_departure: bool = False
    road_type: str = "highway"  # highway, urban, rural


@dataclass
class PerceptionMetrics:
    """Real-time perception metrics"""
    fps: float = 0.0
    processing_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    segmentation_time_ms: float = 0.0
    num_detections: int = 0
    num_tracked_objects: int = 0
    lane_detected: bool = False
    center_offset: float = 0.0
    closest_object_distance: float = float('inf')
    collision_risk: str = "NONE"
    timestamp: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0


@dataclass
class CameraConfig:
    """Camera configuration with intrinsic parameters"""
    device_id: int
    name: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    is_active: bool = False
    position: str = "front"

    # Camera intrinsics
    focal_length: float = 800.0
    principal_point: Tuple[float, float] = (640.0, 360.0)
    distortion_coeffs: np.ndarray = None
    camera_matrix: np.ndarray = None

    # Extrinsic parameters
    height_from_ground: float = 1.2  # meters
    pitch_angle: float = 0.0  # degrees
    yaw_angle: float = 0.0  # degrees


@dataclass
class TrackedObject:
    """Tracked object with Kalman filtering"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    class_id: int
    class_name: str
    confidence: float
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    distance: float = 0.0
    ttc: float = float('inf')
    frames_since_seen: int = 0
    age: int = 0

    # Kalman filter state
    kalman_filter: Any = None
    predicted_position: Tuple[float, float] = (0.0, 0.0)
    predicted_trajectory: List[Tuple[float, float]] = field(default_factory=list)

    # Advanced attributes
    acceleration: float = 0.0
    heading: float = 0.0
    turn_rate: float = 0.0
    behavior: str = "normal"


@dataclass
class OccupancyGrid:
    """Occupancy grid for spatial representation"""
    grid: np.ndarray = None
    resolution: float = 0.1  # meters per cell
    width: int = 400
    height: int = 600
    origin: Tuple[float, float] = (0.0, 0.0)


@dataclass
class SceneUnderstanding:
    """High-level scene understanding"""
    scene_type: str = "highway"  # highway, urban, rural, parking
    weather: str = "clear"  # clear, rain, fog, snow
    time_of_day: str = "day"  # day, night, dawn, dusk
    traffic_density: str = "low"  # low, medium, high
    road_condition: str = "good"  # good, wet, icy, damaged


@dataclass
class DataLogEntry:
    """Entry for data logging"""
    timestamp: float
    frame_id: int
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    lane_info: LaneInfo
    metrics: PerceptionMetrics
    scene_understanding: SceneUnderstanding
    ego_speed: float = 0.0
    ego_position: Tuple[float, float] = (0.0, 0.0)


# ============================================================================
# CAMERA MANAGER WITH CALIBRATION
# ============================================================================

class CameraManager:
    """Manages multiple camera devices with calibration"""

    def __init__(self):
        self.cameras: Dict[int, CameraConfig] = {}
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.frames: Dict[int, np.ndarray] = {}
        self.locks: Dict[int, threading.Lock] = {}
        self.running = False
        self.threads: Dict[int, threading.Thread] = {}
        self.calibration_data: Dict[int, Dict] = {}

    def discover_cameras(self, max_cameras: int = 10) -> List[CameraConfig]:
        """Discover available camera devices"""
        discovered = []
        position_names = ["front", "left", "right", "rear"]

        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

                config = CameraConfig(
                    device_id=i,
                    name=f"Camera {i}",
                    width=width,
                    height=height,
                    fps=fps,
                    position=position_names[len(discovered) % 4] if len(discovered) < 4 else "auxiliary"
                )
                discovered.append(config)
                self.cameras[i] = config
                self.locks[i] = threading.Lock()

                cap.release()
                logger.info(f"Discovered camera {i}: {width}x{height}@{fps}fps")

        return discovered

    def calibrate_camera(self, device_id: int, checkerboard_size: Tuple[int, int] = (9, 6)):
        """Calibrate camera using checkerboard pattern"""
        if device_id not in self.cameras:
            return False

        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        logger.info(f"Camera {device_id} calibration started. Show checkerboard pattern...")

        cap = cv2.VideoCapture(device_id)
        collected = 0

        while collected < 20:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)

                collected += 1
                logger.info(f"Collected {collected}/20 calibration images")
                time.sleep(0.5)

        cap.release()

        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            self.cameras[device_id].camera_matrix = camera_matrix
            self.cameras[device_id].distortion_coeffs = dist_coeffs
            self.calibration_data[device_id] = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
            logger.info(f"Camera {device_id} calibration successful")
            return True
        else:
            logger.error(f"Camera {device_id} calibration failed")
            return False

    def undistort_frame(self, device_id: int, frame: np.ndarray) -> np.ndarray:
        """Undistort frame using calibration data"""
        if device_id in self.calibration_data:
            camera_matrix = self.calibration_data[device_id]['camera_matrix']
            dist_coeffs = self.calibration_data[device_id]['dist_coeffs']
            return cv2.undistort(frame, camera_matrix, dist_coeffs)
        return frame

    def configure_camera(self, device_id: int, width: int = 1280, height: int = 720, fps: int = 30):
        """Configure camera settings"""
        if device_id in self.captures and self.captures[device_id].isOpened():
            cap = self.captures[device_id]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if device_id in self.cameras:
                self.cameras[device_id].width = width
                self.cameras[device_id].height = height
                self.cameras[device_id].fps = fps

    def start_camera(self, device_id: int) -> bool:
        """Start capturing from a specific camera"""
        if device_id not in self.cameras:
            return False

        try:
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                config = self.cameras[device_id]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                cap.set(cv2.CAP_PROP_FPS, config.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                self.captures[device_id] = cap
                self.cameras[device_id].is_active = True

                self.running = True
                thread = threading.Thread(target=self._capture_loop, args=(device_id,), daemon=True)
                thread.start()
                self.threads[device_id] = thread

                logger.info(f"Started camera {device_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to start camera {device_id}: {e}")

        return False

    def stop_camera(self, device_id: int):
        """Stop capturing from a specific camera"""
        if device_id in self.captures:
            self.cameras[device_id].is_active = False

            if self.captures[device_id].isOpened():
                self.captures[device_id].release()
            del self.captures[device_id]

            logger.info(f"Stopped camera {device_id}")

    def stop_all(self):
        """Stop all cameras"""
        self.running = False
        for device_id in list(self.captures.keys()):
            self.stop_camera(device_id)

    def _capture_loop(self, device_id: int):
        """Camera capture thread loop"""
        while self.running and device_id in self.captures:
            cap = self.captures.get(device_id)
            if cap is None or not cap.isOpened():
                break

            ret, frame = cap.read()
            if ret:
                # Undistort if calibrated
                frame = self.undistort_frame(device_id, frame)

                with self.locks[device_id]:
                    self.frames[device_id] = frame
            else:
                time.sleep(0.001)

    def get_frame(self, device_id: int) -> Optional[np.ndarray]:
        """Get the latest frame from a camera"""
        if device_id not in self.locks:
            return None

        with self.locks[device_id]:
            frame = self.frames.get(device_id)
            return frame.copy() if frame is not None else None

    def get_all_frames(self) -> Dict[int, np.ndarray]:
        """Get latest frames from all active cameras"""
        frames = {}
        for device_id in self.cameras:
            if self.cameras[device_id].is_active:
                frame = self.get_frame(device_id)
                if frame is not None:
                    frames[device_id] = frame
        return frames


# ============================================================================
# SEMANTIC SEGMENTATION
# ============================================================================

class SemanticSegmentationEngine:
    """Semantic segmentation for scene understanding"""

    # Cityscapes class labels
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    # Class colors for visualization
    COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

    def __init__(self):
        self.model = None
        self.input_size = (512, 512)
        self.use_deeplab = False

        self._initialize_model()

    def _initialize_model(self):
        """Initialize segmentation model"""
        # Try to use a pre-trained model (would need to download)
        # For now, we'll use a simple color-based segmentation as fallback
        logger.info("Using color-based segmentation (DeepLab model not loaded)")

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic segmentation"""
        if frame is None:
            return None, None

        # Simple color-based segmentation for demo
        seg_mask, class_map = self._color_based_segmentation(frame)

        # Create visualization
        vis = self._visualize_segmentation(frame, seg_mask)

        return seg_mask, vis

    def _color_based_segmentation(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple color-based segmentation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        seg_mask = np.zeros((height, width, 3), dtype=np.uint8)
        class_map = np.zeros((height, width), dtype=np.uint8)

        # Road (dark gray/black at bottom)
        road_mask = np.zeros((height, width), dtype=np.uint8)
        road_mask[int(height * 0.5):, :] = 255
        lower_gray = np.array([0, 0, 0])
        upper_gray = np.array([180, 50, 100])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        road_final = cv2.bitwise_and(road_mask, gray_mask)
        seg_mask[road_final > 0] = self.COLORS[0]
        class_map[road_final > 0] = 0

        # Sky (top third, blue)
        sky_mask = np.zeros((height, width), dtype=np.uint8)
        sky_mask[:int(height * 0.4), :] = 255
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        sky_final = cv2.bitwise_and(sky_mask, blue_mask)
        seg_mask[sky_final > 0] = self.COLORS[10]
        class_map[sky_final > 0] = 10

        # Vegetation (green)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        seg_mask[green_mask > 0] = self.COLORS[8]
        class_map[green_mask > 0] = 8

        return class_map, seg_mask

    def _visualize_segmentation(self, frame: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        """Create segmentation visualization"""
        return cv2.addWeighted(frame, 0.6, seg_mask, 0.4, 0)


# ============================================================================
# DEPTH ESTIMATION
# ============================================================================

class DepthEstimator:
    """Monocular depth estimation"""

    def __init__(self):
        self.model = None
        self.use_midas = False

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from single image"""
        if frame is None:
            return None

        # Simple depth estimation based on vertical position (objects lower = closer)
        height, width = frame.shape[:2]

        # Create depth gradient
        depth_map = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            # Closer at bottom, farther at top
            depth_value = 1.0 - (y / height)
            depth_map[y, :] = depth_value

        # Apply texture-based refinement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_blur = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0

        # Combine
        depth_map = depth_map * (1.0 - edges_blur * 0.3)

        # Normalize
        depth_map = (depth_map * 255).astype(np.uint8)

        return depth_map

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Create colored depth visualization"""
        if depth_map is None:
            return None

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        return depth_colored


# ============================================================================
# KALMAN FILTER FOR TRACKING
# ============================================================================

class KalmanFilter:
    """Kalman filter for object tracking"""

    def __init__(self, dt: float = 0.033):
        """
        State: [x, y, vx, vy, ax, ay]
        Measurement: [x, y]
        """
        self.dt = dt
        self.dim_x = 6  # State dimension
        self.dim_z = 2  # Measurement dimension

        # State vector
        self.x = np.zeros(self.dim_x)

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Covariance matrix
        self.P = np.eye(self.dim_x) * 1000

        # Process noise
        self.Q = np.eye(self.dim_x) * 0.1

        # Measurement noise
        self.R = np.eye(self.dim_z) * 10

    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: np.ndarray):
        """Update with measurement"""
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2]


# ============================================================================
# ADVANCED OBJECT TRACKER WITH KALMAN FILTERING
# ============================================================================

class AdvancedObjectTracker:
    """Advanced multi-object tracker with Kalman filtering"""

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.next_object_id = 0
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.kalman_filters: Dict[int, KalmanFilter] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections"""
        # Predict positions using Kalman filters
        for object_id, kf in self.kalman_filters.items():
            predicted_pos = kf.predict()
            if object_id in self.objects:
                self.objects[object_id].predicted_position = tuple(predicted_pos)

        # If no detections, increment disappeared counter
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].frames_since_seen += 1
                if self.objects[object_id].frames_since_seen > self.max_disappeared:
                    del self.objects[object_id]
                    if object_id in self.kalman_filters:
                        del self.kalman_filters[object_id]
            return list(self.objects.values())

        # Get input centroids and boxes
        input_centroids = []
        input_boxes = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
            input_boxes.append(detection.bbox)

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, detection in enumerate(detections):
                self._register(detection, input_centroids[i])
        else:
            # Match detections to existing objects
            object_ids = list(self.objects.keys())
            object_boxes = [self.objects[oid].bbox for oid in object_ids]

            # Compute IOU matrix
            iou_matrix = np.zeros((len(object_ids), len(input_boxes)))
            for i, obj_box in enumerate(object_boxes):
                for j, inp_box in enumerate(input_boxes):
                    iou_matrix[i, j] = self._compute_iou(obj_box, inp_box)

            # Greedy assignment
            used_rows = set()
            used_cols = set()

            flat_indices = np.argsort(iou_matrix.ravel())[::-1]

            for flat_idx in flat_indices:
                row = flat_idx // len(input_boxes)
                col = flat_idx % len(input_boxes)

                if row in used_rows or col in used_cols:
                    continue

                if iou_matrix[row, col] < self.iou_threshold:
                    break

                object_id = object_ids[row]
                detection = detections[col]
                self._update_object(object_id, detection, input_centroids[col])

                used_rows.add(row)
                used_cols.add(col)

            # Mark unmatched objects
            for row in range(len(object_ids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.objects[object_id].frames_since_seen += 1
                    if self.objects[object_id].frames_since_seen > self.max_disappeared:
                        del self.objects[object_id]
                        if object_id in self.kalman_filters:
                            del self.kalman_filters[object_id]

            # Register unmatched detections
            for col in range(len(input_boxes)):
                if col not in used_cols:
                    self._register(detections[col], input_centroids[col])

        # Predict trajectories
        self._predict_trajectories()

        return list(self.objects.values())

    def _register(self, detection: Detection, centroid: Tuple[float, float]):
        """Register a new object with Kalman filter"""
        tracked = TrackedObject(
            track_id=self.next_object_id,
            bbox=detection.bbox,
            centroid=(int(centroid[0]), int(centroid[1])),
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            distance=detection.distance
        )
        tracked.history.append((int(centroid[0]), int(centroid[1])))

        # Initialize Kalman filter
        kf = KalmanFilter()
        kf.x[:2] = centroid
        self.kalman_filters[self.next_object_id] = kf

        self.objects[self.next_object_id] = tracked
        self.next_object_id += 1

    def _update_object(self, object_id: int, detection: Detection, centroid: Tuple[float, float]):
        """Update existing object with Kalman filter"""
        obj = self.objects[object_id]

        # Update Kalman filter
        if object_id in self.kalman_filters:
            kf = self.kalman_filters[object_id]
            filtered_pos = kf.update(np.array(centroid))
            centroid = tuple(filtered_pos)

        # Calculate velocity and acceleration
        if len(obj.history) > 0:
            prev_centroid = obj.history[-1]
            vx = centroid[0] - prev_centroid[0]
            vy = centroid[1] - prev_centroid[1]

            # Smooth velocity
            alpha = 0.3
            new_vx = alpha * vx + (1 - alpha) * obj.velocity[0]
            new_vy = alpha * vy + (1 - alpha) * obj.velocity[1]
            obj.velocity = (new_vx, new_vy)

            # Calculate acceleration
            if len(obj.history) > 1:
                prev_vel = np.linalg.norm(obj.velocity)
                curr_vel = np.linalg.norm((vx, vy))
                obj.acceleration = (curr_vel - prev_vel) * 30  # Approximate fps

            # Calculate heading
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                obj.heading = np.arctan2(vy, vx) * 180 / np.pi

        obj.bbox = detection.bbox
        obj.centroid = (int(centroid[0]), int(centroid[1]))
        obj.confidence = detection.confidence
        obj.history.append((int(centroid[0]), int(centroid[1])))
        obj.frames_since_seen = 0
        obj.age += 1
        obj.distance = detection.distance

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _predict_trajectories(self, prediction_steps: int = 10):
        """Predict future trajectories for tracked objects"""
        for object_id, obj in self.objects.items():
            if object_id not in self.kalman_filters:
                continue

            kf = self.kalman_filters[object_id]
            trajectory = []

            # Save current state
            saved_state = kf.x.copy()
            saved_cov = kf.P.copy()

            # Predict future positions
            for _ in range(prediction_steps):
                pred_pos = kf.predict()
                trajectory.append((int(pred_pos[0]), int(pred_pos[1])))

            # Restore state
            kf.x = saved_state
            kf.P = saved_cov

            obj.predicted_trajectory = trajectory

    def reset(self):
        """Reset tracker"""
        self.objects.clear()
        self.kalman_filters.clear()
        self.next_object_id = 0


# ============================================================================
# OCCUPANCY GRID MAPPER
# ============================================================================

class OccupancyGridMapper:
    """Create occupancy grid from detections"""

    def __init__(self, width: int = 400, height: int = 600, resolution: float = 0.1):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.grid = np.zeros((height, width), dtype=np.float32)

    def update(self, tracked_objects: List[TrackedObject], ego_position: Tuple[float, float] = (0, 0)):
        """Update occupancy grid with tracked objects"""
        # Decay existing occupancy
        self.grid *= 0.95

        # Add tracked objects
        for obj in tracked_objects:
            # Convert world coordinates to grid coordinates
            grid_x, grid_y = self._world_to_grid(obj.distance, 0, ego_position)

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                # Mark as occupied
                self.grid[grid_y, grid_x] = min(1.0, self.grid[grid_y, grid_x] + 0.8)

                # Add uncertainty
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist > 0:
                                self.grid[ny, nx] = min(1.0, self.grid[ny, nx] + 0.3 / dist)

    def _world_to_grid(self, x: float, y: float, ego_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - ego_pos[0]) / self.resolution + self.width / 2)
        grid_y = int(self.height - (y - ego_pos[1]) / self.resolution)
        return grid_x, grid_y

    def visualize(self) -> np.ndarray:
        """Create visualization of occupancy grid"""
        vis = (self.grid * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        return vis

    def get_free_space(self, threshold: float = 0.5) -> np.ndarray:
        """Get free space mask"""
        return (self.grid < threshold).astype(np.uint8) * 255


# Continue in next part due to length...
