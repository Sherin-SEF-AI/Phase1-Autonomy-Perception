#!/usr/bin/env python3
"""
Advanced Autonomous Vehicle Perception System
Level 1+ Autonomy - Full Perception Stack

Features:
- Multi-camera management (4 cameras)
- Real-time object detection (YOLOv8)
- Lane detection with polynomial curve fitting
- Multi-object tracking (Centroid + IOU)
- Collision warning system (TTC-based)
- Bird's eye view transformation
- Traffic sign detection
- Distance estimation
- Speed estimation
- Comprehensive dashboard
- Recording capability

Author: DeepMost AI Perception Team
Version: 1.0.0
"""

import wx
import wx.lib.newevent
import cv2
import numpy as np
import threading
import queue
import time
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import deque, OrderedDict
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ADAS_Perception')

# Custom events for thread-safe GUI updates
FrameUpdateEvent, EVT_FRAME_UPDATE = wx.lib.newevent.NewEvent()
MetricsUpdateEvent, EVT_METRICS_UPDATE = wx.lib.newevent.NewEvent()
AlertEvent, EVT_ALERT = wx.lib.newevent.NewEvent()


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
    

@dataclass
class Detection:
    """Single object detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    distance: float = 0.0
    velocity: float = 0.0
    track_id: int = -1
    ttc: float = float('inf')  # Time to collision


@dataclass
class LaneInfo:
    """Lane detection results"""
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    center_offset: float = 0.0  # Offset from lane center in meters
    curvature: float = 0.0  # Lane curvature radius
    lane_width: float = 3.7  # Standard lane width in meters
    confidence: float = 0.0
    departure_warning: bool = False


@dataclass
class PerceptionMetrics:
    """Real-time perception metrics"""
    fps: float = 0.0
    processing_time_ms: float = 0.0
    num_detections: int = 0
    num_tracked_objects: int = 0
    lane_detected: bool = False
    center_offset: float = 0.0
    closest_object_distance: float = float('inf')
    collision_risk: str = "NONE"
    timestamp: float = 0.0


@dataclass
class CameraConfig:
    """Camera configuration"""
    device_id: int
    name: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    is_active: bool = False
    position: str = "front"  # front, left, right, rear


@dataclass
class TrackedObject:
    """Tracked object with history"""
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


# ============================================================================
# CAMERA MANAGER
# ============================================================================

class CameraManager:
    """Manages multiple camera devices"""
    
    def __init__(self):
        self.cameras: Dict[int, CameraConfig] = {}
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.frames: Dict[int, np.ndarray] = {}
        self.locks: Dict[int, threading.Lock] = {}
        self.running = False
        self.threads: Dict[int, threading.Thread] = {}
        
    def discover_cameras(self, max_cameras: int = 10) -> List[CameraConfig]:
        """Discover available camera devices"""
        discovered = []
        position_names = ["front", "left", "right", "rear"]
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                
                # Try to get camera name
                name = f"Camera {i}"
                
                config = CameraConfig(
                    device_id=i,
                    name=name,
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
    
    def configure_camera(self, device_id: int, width: int = 1280, height: int = 720, fps: int = 30):
        """Configure camera settings"""
        if device_id in self.captures and self.captures[device_id].isOpened():
            cap = self.captures[device_id]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Update config
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
                
                # Start capture thread
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
# OBJECT DETECTOR (YOLOv8)
# ============================================================================

class ObjectDetector:
    """YOLO-based object detection"""
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Classes relevant to driving
    DRIVING_CLASSES = {0, 1, 2, 3, 5, 7, 9, 11}  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.net = None
        self.use_ultralytics = False
        self.input_size = (640, 640)
        
        self._initialize_model(model_path)
        
    def _initialize_model(self, model_path: str):
        """Initialize YOLO model"""
        # Try ultralytics first
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.use_ultralytics = True
            logger.info("Initialized YOLOv8 with ultralytics")
            return
        except ImportError:
            logger.warning("ultralytics not found, trying OpenCV DNN")
        except Exception as e:
            logger.warning(f"Failed to load ultralytics model: {e}")
        
        # Fallback to OpenCV DNN with YOLOv4-tiny
        try:
            weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
            config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
            
            weights_path = Path.home() / ".cache" / "adas" / "yolov4-tiny.weights"
            config_path = Path.home() / ".cache" / "adas" / "yolov4-tiny.cfg"
            
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not weights_path.exists():
                logger.info("Downloading YOLOv4-tiny weights...")
                import urllib.request
                urllib.request.urlretrieve(weights_url, weights_path)
                
            if not config_path.exists():
                logger.info("Downloading YOLOv4-tiny config...")
                import urllib.request
                urllib.request.urlretrieve(config_url, config_path)
            
            self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            logger.info("Initialized YOLOv4-tiny with OpenCV DNN")
            
        except Exception as e:
            logger.error(f"Failed to initialize any YOLO model: {e}")
            logger.info("Object detection will be disabled")
    
    def detect(self, frame: np.ndarray, filter_driving_classes: bool = True) -> List[Detection]:
        """Detect objects in frame"""
        if frame is None:
            return []
            
        detections = []
        
        if self.use_ultralytics and self.model is not None:
            detections = self._detect_ultralytics(frame, filter_driving_classes)
        elif self.net is not None:
            detections = self._detect_opencv(frame, filter_driving_classes)
            
        return detections
    
    def _detect_ultralytics(self, frame: np.ndarray, filter_driving_classes: bool) -> List[Detection]:
        """Detection using ultralytics YOLOv8"""
        detections = []
        
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    if filter_driving_classes and class_id not in self.DRIVING_CLASSES:
                        continue
                        
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else "unknown"
                    )
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"Ultralytics detection error: {e}")
            
        return detections
    
    def _detect_opencv(self, frame: np.ndarray, filter_driving_classes: bool) -> List[Detection]:
        """Detection using OpenCV DNN"""
        detections = []
        height, width = frame.shape[:2]
        
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        if filter_driving_classes and class_id not in self.DRIVING_CLASSES:
                            continue
                            
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, w, h = boxes[idx]
                
                detection = Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidences[idx],
                    class_id=class_ids[idx],
                    class_name=self.COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(self.COCO_CLASSES) else "unknown"
                )
                detections.append(detection)
                
        except Exception as e:
            logger.error(f"OpenCV DNN detection error: {e}")
            
        return detections


# ============================================================================
# OBJECT TRACKER
# ============================================================================

class ObjectTracker:
    """Multi-object tracker using centroid + IOU tracking"""
    
    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.next_object_id = 0
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections"""
        # If no detections, increment disappeared counter for all objects
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].frames_since_seen += 1
                if self.objects[object_id].frames_since_seen > self.max_disappeared:
                    del self.objects[object_id]
            return list(self.objects.values())
        
        # Get input centroids and boxes
        input_centroids = []
        input_boxes = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))
            input_boxes.append(detection.bbox)
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for i, detection in enumerate(detections):
                self._register(detection, input_centroids[i])
        else:
            # Match detections to existing objects using IOU
            object_ids = list(self.objects.keys())
            object_boxes = [self.objects[oid].bbox for oid in object_ids]
            
            # Compute IOU matrix
            iou_matrix = np.zeros((len(object_ids), len(input_boxes)))
            for i, obj_box in enumerate(object_boxes):
                for j, inp_box in enumerate(input_boxes):
                    iou_matrix[i, j] = self._compute_iou(obj_box, inp_box)
            
            # Hungarian-like greedy assignment
            used_rows = set()
            used_cols = set()
            
            # Sort by IOU (highest first)
            flat_indices = np.argsort(iou_matrix.ravel())[::-1]
            
            for flat_idx in flat_indices:
                row = flat_idx // len(input_boxes)
                col = flat_idx % len(input_boxes)
                
                if row in used_rows or col in used_cols:
                    continue
                    
                if iou_matrix[row, col] < self.iou_threshold:
                    break
                    
                # Update matched object
                object_id = object_ids[row]
                detection = detections[col]
                self._update_object(object_id, detection, input_centroids[col])
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Mark unmatched objects as disappeared
            for row in range(len(object_ids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.objects[object_id].frames_since_seen += 1
                    if self.objects[object_id].frames_since_seen > self.max_disappeared:
                        del self.objects[object_id]
            
            # Register unmatched detections as new objects
            for col in range(len(input_boxes)):
                if col not in used_cols:
                    self._register(detections[col], input_centroids[col])
        
        return list(self.objects.values())
    
    def _register(self, detection: Detection, centroid: Tuple[int, int]):
        """Register a new object"""
        tracked = TrackedObject(
            track_id=self.next_object_id,
            bbox=detection.bbox,
            centroid=centroid,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            distance=detection.distance
        )
        tracked.history.append(centroid)
        self.objects[self.next_object_id] = tracked
        self.next_object_id += 1
        
    def _update_object(self, object_id: int, detection: Detection, centroid: Tuple[int, int]):
        """Update existing object"""
        obj = self.objects[object_id]
        
        # Calculate velocity
        if len(obj.history) > 0:
            prev_centroid = obj.history[-1]
            vx = centroid[0] - prev_centroid[0]
            vy = centroid[1] - prev_centroid[1]
            # Smooth velocity
            alpha = 0.3
            obj.velocity = (
                alpha * vx + (1 - alpha) * obj.velocity[0],
                alpha * vy + (1 - alpha) * obj.velocity[1]
            )
        
        obj.bbox = detection.bbox
        obj.centroid = centroid
        obj.confidence = detection.confidence
        obj.history.append(centroid)
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
    
    def reset(self):
        """Reset tracker"""
        self.objects.clear()
        self.next_object_id = 0


# ============================================================================
# LANE DETECTOR
# ============================================================================

class LaneDetector:
    """Advanced lane detection with polynomial fitting"""
    
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
        # Camera calibration parameters (approximate for typical webcam)
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
        # ROI parameters
        self.roi_vertices = None
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, LaneInfo]:
        """Detect lanes in frame"""
        if frame is None:
            return frame, LaneInfo()
            
        height, width = frame.shape[:2]
        
        # Define ROI
        if self.roi_vertices is None:
            self.roi_vertices = np.array([
                [(int(width * 0.1), height),
                 (int(width * 0.4), int(height * 0.6)),
                 (int(width * 0.6), int(height * 0.6)),
                 (int(width * 0.9), height)]
            ], dtype=np.int32)
        
        # Preprocess
        processed = self._preprocess(frame)
        
        # Apply ROI mask
        masked = self._apply_roi(processed)
        
        # Detect lane lines
        lane_info = self._detect_lanes(masked, frame.shape)
        
        # Draw lanes on frame
        result = self._draw_lanes(frame, lane_info)
        
        return result, lane_info
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for lane detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Also use color thresholding for white and yellow lanes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White color mask
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Yellow color mask
        yellow_lower = np.array([15, 50, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine masks
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Combine edge and color detection
        combined = cv2.bitwise_or(edges, color_mask)
        
        return combined
    
    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        """Apply region of interest mask"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(img, mask)
    
    def _detect_lanes(self, binary: np.ndarray, shape: Tuple) -> LaneInfo:
        """Detect lane lines using sliding window"""
        height, width = shape[:2]
        lane_info = LaneInfo()
        
        # Use Hough transform
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )
        
        if lines is None:
            return lane_info
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Filter by slope
            if abs(slope) < 0.3 or abs(slope) > 3:
                continue
            
            if slope < 0:  # Left lane (negative slope due to image coordinates)
                left_lines.append((slope, intercept))
            else:  # Right lane
                right_lines.append((slope, intercept))
        
        # Average the lines
        if left_lines:
            left_avg = np.mean(left_lines, axis=0)
            lane_info.left_lane = self._make_line_points(left_avg, height)
            self.left_fit = left_avg
            self.left_fit_history.append(left_avg)
        elif self.left_fit_history:
            # Use historical fit
            lane_info.left_lane = self._make_line_points(np.mean(self.left_fit_history, axis=0), height)
            
        if right_lines:
            right_avg = np.mean(right_lines, axis=0)
            lane_info.right_lane = self._make_line_points(right_avg, height)
            self.right_fit = right_avg
            self.right_fit_history.append(right_avg)
        elif self.right_fit_history:
            lane_info.right_lane = self._make_line_points(np.mean(self.right_fit_history, axis=0), height)
        
        # Calculate metrics
        if lane_info.left_lane is not None and lane_info.right_lane is not None:
            lane_info.confidence = 1.0
            
            # Calculate center offset
            lane_center = (lane_info.left_lane[0][0] + lane_info.right_lane[0][0]) / 2
            frame_center = width / 2
            lane_info.center_offset = (frame_center - lane_center) * self.xm_per_pix
            
            # Calculate curvature (simplified)
            lane_info.curvature = self._calculate_curvature(lane_info)
            
            # Departure warning
            if abs(lane_info.center_offset) > 0.5:  # More than 0.5m from center
                lane_info.departure_warning = True
        elif lane_info.left_lane is not None or lane_info.right_lane is not None:
            lane_info.confidence = 0.5
            
        return lane_info
    
    def _make_line_points(self, line: Tuple[float, float], height: int) -> np.ndarray:
        """Convert slope-intercept to line points"""
        slope, intercept = line
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([[x1, y1], [x2, y2]])
    
    def _calculate_curvature(self, lane_info: LaneInfo) -> float:
        """Calculate lane curvature"""
        # Simplified curvature calculation
        if lane_info.left_lane is None or lane_info.right_lane is None:
            return 0.0
            
        # Use the x difference at different y levels
        left_dx = lane_info.left_lane[1][0] - lane_info.left_lane[0][0]
        right_dx = lane_info.right_lane[1][0] - lane_info.right_lane[0][0]
        
        avg_dx = (left_dx + right_dx) / 2
        
        # Convert to curvature radius (approximate)
        if abs(avg_dx) < 5:
            return float('inf')  # Straight road
            
        return 1000 / abs(avg_dx)  # Approximate radius
    
    def _draw_lanes(self, frame: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """Draw detected lanes on frame"""
        overlay = frame.copy()
        
        # Draw lane polygon
        if lane_info.left_lane is not None and lane_info.right_lane is not None:
            pts = np.array([
                lane_info.left_lane[0],
                lane_info.left_lane[1],
                lane_info.right_lane[1],
                lane_info.right_lane[0]
            ], dtype=np.int32)
            
            color = (0, 100, 0) if not lane_info.departure_warning else (0, 0, 150)
            cv2.fillPoly(overlay, [pts], color)
            
        # Draw lane lines
        if lane_info.left_lane is not None:
            cv2.line(overlay, tuple(lane_info.left_lane[0]), tuple(lane_info.left_lane[1]), (0, 255, 255), 3)
            
        if lane_info.right_lane is not None:
            cv2.line(overlay, tuple(lane_info.right_lane[0]), tuple(lane_info.right_lane[1]), (0, 255, 255), 3)
        
        # Blend with original
        result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        return result


# ============================================================================
# DISTANCE ESTIMATOR
# ============================================================================

class DistanceEstimator:
    """Estimate distance to detected objects"""
    
    # Reference object heights in meters
    REFERENCE_HEIGHTS = {
        'person': 1.7,
        'car': 1.5,
        'truck': 3.0,
        'bus': 3.2,
        'motorcycle': 1.2,
        'bicycle': 1.0,
        'traffic light': 0.6,
        'stop sign': 0.75
    }
    
    def __init__(self, focal_length: float = 800, sensor_height: float = 720):
        self.focal_length = focal_length
        self.sensor_height = sensor_height
        
    def estimate(self, detections: List[Detection], frame_height: int) -> List[Detection]:
        """Estimate distance for each detection"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            bbox_height = y2 - y1
            
            # Get reference height
            ref_height = self.REFERENCE_HEIGHTS.get(detection.class_name, 1.5)
            
            # Calculate distance using similar triangles
            if bbox_height > 0:
                distance = (ref_height * self.focal_length) / bbox_height
                detection.distance = min(distance, 100)  # Cap at 100m
            else:
                detection.distance = 100
                
        return detections


# ============================================================================
# COLLISION WARNING SYSTEM
# ============================================================================

class CollisionWarningSystem:
    """Time-to-collision based collision warning"""
    
    def __init__(self, warning_ttc: float = 3.0, critical_ttc: float = 1.5):
        self.warning_ttc = warning_ttc
        self.critical_ttc = critical_ttc
        self.ego_speed = 0.0  # m/s (would come from vehicle CAN bus)
        self.previous_distances: Dict[int, deque] = {}
        
    def set_ego_speed(self, speed_kmh: float):
        """Set ego vehicle speed"""
        self.ego_speed = speed_kmh / 3.6  # Convert to m/s
        
    def calculate_ttc(self, tracked_objects: List[TrackedObject], dt: float = 0.033) -> Tuple[List[TrackedObject], AlertLevel]:
        """Calculate time to collision for tracked objects"""
        max_alert = AlertLevel.INFO
        
        for obj in tracked_objects:
            track_id = obj.track_id
            
            # Initialize history if needed
            if track_id not in self.previous_distances:
                self.previous_distances[track_id] = deque(maxlen=10)
                
            self.previous_distances[track_id].append(obj.distance)
            
            # Calculate relative velocity
            if len(self.previous_distances[track_id]) >= 2:
                distances = list(self.previous_distances[track_id])
                rel_velocity = (distances[-2] - distances[-1]) / dt
                
                # Add ego speed
                closing_speed = rel_velocity + self.ego_speed
                
                # Calculate TTC
                if closing_speed > 0.5:  # Approaching
                    ttc = obj.distance / closing_speed
                    obj.ttc = ttc
                    
                    # Determine alert level
                    if ttc < self.critical_ttc:
                        max_alert = AlertLevel.CRITICAL
                    elif ttc < self.warning_ttc and max_alert.value < AlertLevel.DANGER.value:
                        max_alert = AlertLevel.DANGER
                else:
                    obj.ttc = float('inf')
                    
        # Cleanup old tracks
        active_ids = {obj.track_id for obj in tracked_objects}
        for track_id in list(self.previous_distances.keys()):
            if track_id not in active_ids:
                del self.previous_distances[track_id]
                
        return tracked_objects, max_alert


# ============================================================================
# BIRD'S EYE VIEW TRANSFORMER
# ============================================================================

class BirdEyeView:
    """Transform camera view to bird's eye view"""
    
    def __init__(self, src_points: Optional[np.ndarray] = None, dst_points: Optional[np.ndarray] = None):
        self.src_points = src_points
        self.dst_points = dst_points
        self.M = None
        self.M_inv = None
        self.output_size = (400, 600)
        
    def calibrate(self, frame_shape: Tuple[int, int]):
        """Set up perspective transform matrices"""
        height, width = frame_shape[:2]
        
        if self.src_points is None:
            # Default source points (trapezoid in original image)
            self.src_points = np.float32([
                [width * 0.2, height * 0.9],   # Bottom left
                [width * 0.4, height * 0.6],   # Top left
                [width * 0.6, height * 0.6],   # Top right
                [width * 0.8, height * 0.9]    # Bottom right
            ])
            
        if self.dst_points is None:
            # Destination points (rectangle)
            self.dst_points = np.float32([
                [100, self.output_size[1]],
                [100, 0],
                [300, 0],
                [300, self.output_size[1]]
            ])
            
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Transform frame to bird's eye view"""
        if self.M is None:
            self.calibrate(frame.shape)
            
        return cv2.warpPerspective(frame, self.M, self.output_size)
    
    def transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Transform a single point"""
        if self.M is None:
            return point
            
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.M)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))


# ============================================================================
# PERCEPTION ENGINE
# ============================================================================

class PerceptionEngine:
    """Main perception processing engine"""
    
    def __init__(self):
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.lane_detector = LaneDetector()
        self.distance_estimator = DistanceEstimator()
        self.collision_system = CollisionWarningSystem()
        self.bird_eye_view = BirdEyeView()
        
        self.processing_times = deque(maxlen=30)
        self.frame_count = 0
        self.last_time = time.time()
        
    def process_frame(self, frame: np.ndarray, enable_detection: bool = True,
                     enable_lanes: bool = True, enable_tracking: bool = True) -> Tuple[np.ndarray, PerceptionMetrics]:
        """Process a single frame through the perception pipeline"""
        start_time = time.time()
        metrics = PerceptionMetrics()
        
        if frame is None:
            return frame, metrics
            
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Object Detection
        detections = []
        if enable_detection:
            detections = self.detector.detect(frame)
            detections = self.distance_estimator.estimate(detections, height)
            metrics.num_detections = len(detections)
            
        # Object Tracking
        tracked_objects = []
        alert_level = AlertLevel.INFO
        if enable_tracking and detections:
            tracked_objects = self.tracker.update(detections)
            tracked_objects, alert_level = self.collision_system.calculate_ttc(tracked_objects)
            metrics.num_tracked_objects = len(tracked_objects)
            
            # Find closest object
            if tracked_objects:
                min_dist = min(obj.distance for obj in tracked_objects)
                metrics.closest_object_distance = min_dist
                
        # Lane Detection
        lane_info = LaneInfo()
        if enable_lanes:
            result, lane_info = self.lane_detector.detect(result)
            metrics.lane_detected = lane_info.confidence > 0
            metrics.center_offset = lane_info.center_offset
            
        # Draw detections
        result = self._draw_detections(result, tracked_objects, alert_level)
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        processing_time = (current_time - start_time) * 1000
        self.processing_times.append(processing_time)
        
        if current_time - self.last_time >= 1.0:
            metrics.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        else:
            metrics.fps = len(self.processing_times) / (sum(self.processing_times) / 1000) if self.processing_times else 0
            
        metrics.processing_time_ms = np.mean(self.processing_times) if self.processing_times else 0
        metrics.collision_risk = alert_level.name
        metrics.timestamp = current_time
        
        return result, metrics
    
    def _draw_detections(self, frame: np.ndarray, tracked_objects: List[TrackedObject], alert_level: AlertLevel) -> np.ndarray:
        """Draw detection boxes and info on frame"""
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Color based on TTC
            if obj.ttc < 1.5:
                color = (0, 0, 255)  # Red - Critical
            elif obj.ttc < 3.0:
                color = (0, 165, 255)  # Orange - Warning
            else:
                color = (0, 255, 0)  # Green - Safe
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track trail
            if len(obj.history) > 1:
                points = np.array(list(obj.history), dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
            
            # Draw label
            label = f"ID:{obj.track_id} {obj.class_name}"
            distance_label = f"{obj.distance:.1f}m"
            ttc_label = f"TTC:{obj.ttc:.1f}s" if obj.ttc < 10 else ""
            
            # Background for text
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 40), (x1 + max(w1, 80), y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, distance_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if ttc_label:
                cv2.putText(frame, ttc_label, (x1 + 50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Draw alert overlay
        if alert_level == AlertLevel.CRITICAL:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
            cv2.putText(frame, "COLLISION WARNING!", (frame.shape[1]//2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif alert_level == AlertLevel.DANGER:
            cv2.putText(frame, "CAUTION: Close Object", (frame.shape[1]//2 - 120, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                       
        return frame
    
    def get_bird_eye_view(self, frame: np.ndarray, tracked_objects: List[TrackedObject] = None) -> np.ndarray:
        """Generate bird's eye view visualization"""
        bev = self.bird_eye_view.transform(frame)
        
        # Draw tracked objects on BEV
        if tracked_objects:
            for obj in tracked_objects:
                # Transform centroid
                centroid = obj.centroid
                bev_point = self.bird_eye_view.transform_point(centroid)
                
                if 0 <= bev_point[0] < bev.shape[1] and 0 <= bev_point[1] < bev.shape[0]:
                    cv2.circle(bev, bev_point, 10, (0, 255, 0), -1)
                    cv2.putText(bev, f"ID:{obj.track_id}", (bev_point[0] - 15, bev_point[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                               
        return bev


# ============================================================================
# VIDEO RECORDER
# ============================================================================

class VideoRecorder:
    """Record video with perception overlay"""
    
    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.output_path = None
        self.frame_count = 0
        
    def start(self, output_path: str, frame_size: Tuple[int, int], fps: float = 30.0):
        """Start recording"""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.is_recording = True
        self.frame_count = 0
        logger.info(f"Started recording to {output_path}")
        
    def write_frame(self, frame: np.ndarray):
        """Write a frame"""
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1
            
    def stop(self):
        """Stop recording"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        logger.info(f"Stopped recording. Saved {self.frame_count} frames to {self.output_path}")


# ============================================================================
# WX PYTHON GUI COMPONENTS
# ============================================================================

class CameraPanel(wx.Panel):
    """Panel displaying camera feed"""
    
    def __init__(self, parent, camera_id: int = 0):
        super().__init__(parent)
        self.camera_id = camera_id
        self.bitmap = None
        self.frame_size = (640, 480)
        
        self.SetBackgroundColour(wx.Colour(30, 30, 30))
        self.SetMinSize(wx.Size(320, 240))
        
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        
    def update_frame(self, frame: np.ndarray):
        """Update displayed frame"""
        if frame is None:
            return
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit panel
        panel_size = self.GetSize()
        if panel_size.width > 0 and panel_size.height > 0:
            # Maintain aspect ratio
            h, w = frame_rgb.shape[:2]
            aspect = w / h
            
            new_width = panel_size.width
            new_height = int(new_width / aspect)
            
            if new_height > panel_size.height:
                new_height = panel_size.height
                new_width = int(new_height * aspect)
                
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            self.frame_size = (new_width, new_height)
            
        # Convert to wx.Bitmap
        h, w = frame_rgb.shape[:2]
        self.bitmap = wx.Bitmap.FromBuffer(w, h, frame_rgb)
        
        self.Refresh()
        
    def on_paint(self, event):
        """Paint event handler"""
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(30, 30, 30)))
        dc.Clear()
        
        if self.bitmap is not None:
            # Center the image
            panel_size = self.GetSize()
            x = (panel_size.width - self.frame_size[0]) // 2
            y = (panel_size.height - self.frame_size[1]) // 2
            dc.DrawBitmap(self.bitmap, x, y)
        else:
            # Draw placeholder text
            dc.SetTextForeground(wx.Colour(100, 100, 100))
            dc.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            text = f"Camera {self.camera_id}"
            tw, th = dc.GetTextExtent(text)
            dc.DrawText(text, (self.GetSize().width - tw) // 2, (self.GetSize().height - th) // 2)
            
    def on_size(self, event):
        """Handle resize"""
        self.Refresh()
        event.Skip()


class DashboardPanel(wx.Panel):
    """Dashboard showing real-time metrics"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(wx.Colour(25, 25, 35))
        
        self.metrics = PerceptionMetrics()
        self._create_ui()
        
    def _create_ui(self):
        """Create dashboard UI"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="PERCEPTION DASHBOARD")
        title.SetForegroundColour(wx.Colour(0, 200, 255))
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        # Metrics grid
        grid = wx.FlexGridSizer(rows=8, cols=2, vgap=8, hgap=15)
        
        self.labels = {}
        metrics_config = [
            ("FPS:", "fps", "0.0"),
            ("Processing:", "processing", "0ms"),
            ("Detections:", "detections", "0"),
            ("Tracked:", "tracked", "0"),
            ("Lane Status:", "lane", "NOT DETECTED"),
            ("Offset:", "offset", "0.00m"),
            ("Closest Object:", "closest", "---"),
            ("Risk Level:", "risk", "NONE")
        ]
        
        for label_text, key, default in metrics_config:
            label = wx.StaticText(self, label=label_text)
            label.SetForegroundColour(wx.Colour(150, 150, 150))
            label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            
            value = wx.StaticText(self, label=default)
            value.SetForegroundColour(wx.Colour(255, 255, 255))
            value.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            
            self.labels[key] = value
            
            grid.Add(label, 0, wx.ALIGN_LEFT)
            grid.Add(value, 0, wx.ALIGN_RIGHT)
            
        main_sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 10)
        
        # Alert panel
        self.alert_panel = wx.Panel(self)
        self.alert_panel.SetBackgroundColour(wx.Colour(40, 40, 50))
        alert_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.alert_text = wx.StaticText(self.alert_panel, label="SYSTEM READY")
        self.alert_text.SetForegroundColour(wx.Colour(0, 255, 0))
        self.alert_text.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        alert_sizer.Add(self.alert_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        self.alert_panel.SetSizer(alert_sizer)
        main_sizer.Add(self.alert_panel, 0, wx.ALL | wx.EXPAND, 5)
        
        self.SetSizer(main_sizer)
        
    def update_metrics(self, metrics: PerceptionMetrics):
        """Update displayed metrics"""
        self.metrics = metrics
        
        self.labels["fps"].SetLabel(f"{metrics.fps:.1f}")
        self.labels["processing"].SetLabel(f"{metrics.processing_time_ms:.1f}ms")
        self.labels["detections"].SetLabel(str(metrics.num_detections))
        self.labels["tracked"].SetLabel(str(metrics.num_tracked_objects))
        
        # Lane status
        if metrics.lane_detected:
            self.labels["lane"].SetLabel("DETECTED")
            self.labels["lane"].SetForegroundColour(wx.Colour(0, 255, 0))
        else:
            self.labels["lane"].SetLabel("NOT DETECTED")
            self.labels["lane"].SetForegroundColour(wx.Colour(255, 165, 0))
            
        # Offset
        offset_str = f"{metrics.center_offset:.2f}m"
        self.labels["offset"].SetLabel(offset_str)
        if abs(metrics.center_offset) > 0.5:
            self.labels["offset"].SetForegroundColour(wx.Colour(255, 165, 0))
        else:
            self.labels["offset"].SetForegroundColour(wx.Colour(255, 255, 255))
            
        # Closest object
        if metrics.closest_object_distance < 100:
            self.labels["closest"].SetLabel(f"{metrics.closest_object_distance:.1f}m")
        else:
            self.labels["closest"].SetLabel("---")
            
        # Risk level
        risk_colors = {
            "NONE": wx.Colour(0, 255, 0),
            "INFO": wx.Colour(0, 200, 255),
            "WARNING": wx.Colour(255, 255, 0),
            "DANGER": wx.Colour(255, 165, 0),
            "CRITICAL": wx.Colour(255, 0, 0)
        }
        self.labels["risk"].SetLabel(metrics.collision_risk)
        self.labels["risk"].SetForegroundColour(risk_colors.get(metrics.collision_risk, wx.Colour(255, 255, 255)))
        
        # Update alert panel
        if metrics.collision_risk == "CRITICAL":
            self.alert_panel.SetBackgroundColour(wx.Colour(100, 0, 0))
            self.alert_text.SetLabel("⚠ COLLISION WARNING ⚠")
            self.alert_text.SetForegroundColour(wx.Colour(255, 255, 255))
        elif metrics.collision_risk == "DANGER":
            self.alert_panel.SetBackgroundColour(wx.Colour(100, 50, 0))
            self.alert_text.SetLabel("CAUTION: Close Objects")
            self.alert_text.SetForegroundColour(wx.Colour(255, 200, 0))
        else:
            self.alert_panel.SetBackgroundColour(wx.Colour(40, 40, 50))
            self.alert_text.SetLabel("SYSTEM ACTIVE")
            self.alert_text.SetForegroundColour(wx.Colour(0, 255, 0))
            
        self.alert_panel.Refresh()
        self.Layout()


class ControlPanel(wx.Panel):
    """Control panel for system settings"""
    
    def __init__(self, parent, on_setting_changed):
        super().__init__(parent)
        self.on_setting_changed = on_setting_changed
        self.SetBackgroundColour(wx.Colour(35, 35, 45))
        
        self._create_ui()
        
    def _create_ui(self):
        """Create control panel UI"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="CONTROLS")
        title.SetForegroundColour(wx.Colour(0, 200, 255))
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        # Feature toggles
        self.toggles = {}
        toggle_config = [
            ("enable_detection", "Object Detection", True),
            ("enable_tracking", "Object Tracking", True),
            ("enable_lanes", "Lane Detection", True),
            ("show_bev", "Bird's Eye View", False)
        ]
        
        for key, label, default in toggle_config:
            cb = wx.CheckBox(self, label=label)
            cb.SetValue(default)
            cb.SetForegroundColour(wx.Colour(200, 200, 200))
            cb.Bind(wx.EVT_CHECKBOX, lambda e, k=key: self._on_toggle(k, e))
            self.toggles[key] = cb
            main_sizer.Add(cb, 0, wx.ALL | wx.EXPAND, 5)
            
        main_sizer.AddSpacer(10)
        
        # Confidence threshold slider
        threshold_label = wx.StaticText(self, label="Detection Threshold:")
        threshold_label.SetForegroundColour(wx.Colour(200, 200, 200))
        main_sizer.Add(threshold_label, 0, wx.LEFT | wx.TOP, 5)
        
        self.threshold_slider = wx.Slider(self, value=40, minValue=10, maxValue=90,
                                          style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.threshold_slider.Bind(wx.EVT_SLIDER, self._on_threshold_change)
        main_sizer.Add(self.threshold_slider, 0, wx.ALL | wx.EXPAND, 5)
        
        main_sizer.AddSpacer(10)
        
        # Ego speed input
        speed_label = wx.StaticText(self, label="Ego Speed (km/h):")
        speed_label.SetForegroundColour(wx.Colour(200, 200, 200))
        main_sizer.Add(speed_label, 0, wx.LEFT | wx.TOP, 5)
        
        self.speed_spin = wx.SpinCtrl(self, value="30", min=0, max=200)
        self.speed_spin.Bind(wx.EVT_SPINCTRL, self._on_speed_change)
        main_sizer.Add(self.speed_spin, 0, wx.ALL | wx.EXPAND, 5)
        
        self.SetSizer(main_sizer)
        
    def _on_toggle(self, key: str, event):
        """Handle toggle change"""
        self.on_setting_changed(key, event.IsChecked())
        
    def _on_threshold_change(self, event):
        """Handle threshold change"""
        self.on_setting_changed("confidence_threshold", self.threshold_slider.GetValue() / 100.0)
        
    def _on_speed_change(self, event):
        """Handle speed change"""
        self.on_setting_changed("ego_speed", self.speed_spin.GetValue())
        
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return {
            "enable_detection": self.toggles["enable_detection"].GetValue(),
            "enable_tracking": self.toggles["enable_tracking"].GetValue(),
            "enable_lanes": self.toggles["enable_lanes"].GetValue(),
            "show_bev": self.toggles["show_bev"].GetValue(),
            "confidence_threshold": self.threshold_slider.GetValue() / 100.0,
            "ego_speed": self.speed_spin.GetValue()
        }


class CameraSelectionDialog(wx.Dialog):
    """Dialog for selecting and configuring cameras"""
    
    def __init__(self, parent, camera_manager: CameraManager):
        super().__init__(parent, title="Camera Configuration", size=(500, 400))
        self.camera_manager = camera_manager
        self.selected_cameras = []
        
        self._create_ui()
        
    def _create_ui(self):
        """Create dialog UI"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Instructions
        instr = wx.StaticText(self, label="Select cameras to use (up to 4):")
        main_sizer.Add(instr, 0, wx.ALL, 10)
        
        # Camera list
        self.camera_list = wx.CheckListBox(self)
        
        cameras = self.camera_manager.discover_cameras()
        for cam in cameras:
            self.camera_list.Append(f"Camera {cam.device_id}: {cam.width}x{cam.height}@{cam.fps}fps ({cam.position})")
            
        main_sizer.Add(self.camera_list, 1, wx.ALL | wx.EXPAND, 10)
        
        # Resolution selection
        res_sizer = wx.BoxSizer(wx.HORIZONTAL)
        res_label = wx.StaticText(self, label="Resolution:")
        self.res_choice = wx.Choice(self, choices=["1280x720", "640x480", "1920x1080", "800x600"])
        self.res_choice.SetSelection(0)
        res_sizer.Add(res_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        res_sizer.Add(self.res_choice, 0, wx.ALL, 5)
        main_sizer.Add(res_sizer, 0, wx.ALL, 5)
        
        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(self, wx.ID_OK)
        cancel_btn = wx.Button(self, wx.ID_CANCEL)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 10)
        
        self.SetSizer(main_sizer)
        
    def get_selection(self) -> List[Tuple[int, int, int]]:
        """Get selected cameras with resolution"""
        selection = []
        checked = self.camera_list.GetCheckedItems()
        
        res_str = self.res_choice.GetStringSelection()
        width, height = map(int, res_str.split('x'))
        
        cameras = list(self.camera_manager.cameras.values())
        for idx in checked[:4]:  # Limit to 4 cameras
            if idx < len(cameras):
                selection.append((cameras[idx].device_id, width, height))
                
        return selection


# ============================================================================
# MAIN APPLICATION FRAME
# ============================================================================

class MainFrame(wx.Frame):
    """Main application frame"""
    
    def __init__(self):
        super().__init__(None, title="ADAS Perception System - Level 1+ Autonomy", size=(1600, 900))
        
        # Initialize components
        self.camera_manager = CameraManager()
        self.perception_engine = PerceptionEngine()
        self.video_recorder = VideoRecorder()
        
        # State
        self.active_cameras: List[int] = []
        self.primary_camera: int = 0
        self.running = False
        self.settings = {
            "enable_detection": True,
            "enable_tracking": True,
            "enable_lanes": True,
            "show_bev": False,
            "confidence_threshold": 0.4,
            "ego_speed": 30
        }
        
        # Frame queue for processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        self._create_ui()
        self._create_menu()
        self._bind_events()
        
        # Start with camera selection
        wx.CallAfter(self._show_camera_dialog)
        
    def _create_ui(self):
        """Create main UI"""
        self.SetBackgroundColour(wx.Colour(20, 20, 25))
        
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left side - Camera views
        left_panel = wx.Panel(self)
        left_panel.SetBackgroundColour(wx.Colour(20, 20, 25))
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Main camera view
        self.main_camera_panel = CameraPanel(left_panel, 0)
        self.main_camera_panel.SetMinSize(wx.Size(800, 600))
        left_sizer.Add(self.main_camera_panel, 3, wx.ALL | wx.EXPAND, 5)
        
        # Secondary camera views (horizontal)
        secondary_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.secondary_panels: List[CameraPanel] = []
        for i in range(3):
            panel = CameraPanel(left_panel, i + 1)
            panel.SetMinSize(wx.Size(250, 180))
            self.secondary_panels.append(panel)
            secondary_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 3)
            
        left_sizer.Add(secondary_sizer, 1, wx.ALL | wx.EXPAND, 5)
        left_panel.SetSizer(left_sizer)
        
        main_sizer.Add(left_panel, 3, wx.ALL | wx.EXPAND, 5)
        
        # Right side - Dashboard and controls
        right_panel = wx.Panel(self)
        right_panel.SetBackgroundColour(wx.Colour(25, 25, 30))
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Dashboard
        self.dashboard = DashboardPanel(right_panel)
        right_sizer.Add(self.dashboard, 1, wx.ALL | wx.EXPAND, 5)
        
        # Bird's eye view
        self.bev_panel = CameraPanel(right_panel, -1)
        self.bev_panel.SetMinSize(wx.Size(300, 250))
        right_sizer.Add(self.bev_panel, 1, wx.ALL | wx.EXPAND, 5)
        
        # Control panel
        self.control_panel = ControlPanel(right_panel, self._on_setting_changed)
        right_sizer.Add(self.control_panel, 1, wx.ALL | wx.EXPAND, 5)
        
        # Action buttons
        btn_panel = wx.Panel(right_panel)
        btn_panel.SetBackgroundColour(wx.Colour(35, 35, 45))
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.start_btn = wx.Button(btn_panel, label="▶ START")
        self.start_btn.SetBackgroundColour(wx.Colour(0, 120, 0))
        self.start_btn.SetForegroundColour(wx.Colour(255, 255, 255))
        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start_stop)
        btn_sizer.Add(self.start_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        self.record_btn = wx.Button(btn_panel, label="⏺ RECORD")
        self.record_btn.Bind(wx.EVT_BUTTON, self._on_record)
        btn_sizer.Add(self.record_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        btn_panel.SetSizer(btn_sizer)
        right_sizer.Add(btn_panel, 0, wx.ALL | wx.EXPAND, 5)
        
        right_panel.SetSizer(right_sizer)
        main_sizer.Add(right_panel, 1, wx.ALL | wx.EXPAND, 5)
        
        self.SetSizer(main_sizer)
        
        # Status bar
        self.CreateStatusBar(3)
        self.SetStatusWidths([-2, -1, -1])
        self.SetStatusText("Ready", 0)
        self.SetStatusText("Cameras: 0", 1)
        self.SetStatusText("", 2)
        
    def _create_menu(self):
        """Create menu bar"""
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_OPEN, "Open Video...\tCtrl+O")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "Exit\tCtrl+Q")
        menubar.Append(file_menu, "&File")
        
        # Camera menu
        camera_menu = wx.Menu()
        self.camera_config_item = camera_menu.Append(wx.ID_ANY, "Configure Cameras...\tCtrl+K")
        camera_menu.AppendSeparator()
        self.switch_camera_menu = wx.Menu()
        camera_menu.AppendSubMenu(self.switch_camera_menu, "Switch Primary Camera")
        menubar.Append(camera_menu, "&Camera")
        
        # View menu
        view_menu = wx.Menu()
        view_menu.AppendCheckItem(wx.ID_ANY, "Show Bird's Eye View")
        view_menu.AppendCheckItem(wx.ID_ANY, "Show Debug Info")
        menubar.Append(view_menu, "&View")
        
        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT, "About")
        menubar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menubar)
        
        # Bind menu events
        self.Bind(wx.EVT_MENU, self._on_open_video, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self._on_exit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self._on_camera_config, self.camera_config_item)
        self.Bind(wx.EVT_MENU, self._on_about, id=wx.ID_ABOUT)
        
    def _bind_events(self):
        """Bind events"""
        self.Bind(EVT_FRAME_UPDATE, self._on_frame_update)
        self.Bind(EVT_METRICS_UPDATE, self._on_metrics_update)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        
    def _show_camera_dialog(self):
        """Show camera configuration dialog"""
        dialog = CameraSelectionDialog(self, self.camera_manager)
        if dialog.ShowModal() == wx.ID_OK:
            selection = dialog.get_selection()
            if selection:
                self._setup_cameras(selection)
            else:
                wx.MessageBox("No cameras selected. Please select at least one camera.",
                             "Warning", wx.OK | wx.ICON_WARNING)
        dialog.Destroy()
        
    def _setup_cameras(self, camera_configs: List[Tuple[int, int, int]]):
        """Setup selected cameras"""
        # Stop existing cameras
        self.camera_manager.stop_all()
        self.active_cameras.clear()
        
        # Start new cameras
        for device_id, width, height in camera_configs:
            self.camera_manager.configure_camera(device_id, width, height)
            if self.camera_manager.start_camera(device_id):
                self.active_cameras.append(device_id)
                
        if self.active_cameras:
            self.primary_camera = self.active_cameras[0]
            self._update_camera_menu()
            self.SetStatusText(f"Cameras: {len(self.active_cameras)}", 1)
            logger.info(f"Setup {len(self.active_cameras)} cameras")
            
    def _update_camera_menu(self):
        """Update camera switching menu"""
        # Clear existing items
        for item in self.switch_camera_menu.GetMenuItems():
            self.switch_camera_menu.Delete(item)
            
        # Add camera items
        for cam_id in self.active_cameras:
            item = self.switch_camera_menu.AppendRadioItem(wx.ID_ANY, f"Camera {cam_id}")
            self.Bind(wx.EVT_MENU, lambda e, cid=cam_id: self._switch_primary_camera(cid), item)
            if cam_id == self.primary_camera:
                item.Check(True)
                
    def _switch_primary_camera(self, camera_id: int):
        """Switch primary camera view"""
        self.primary_camera = camera_id
        logger.info(f"Switched primary camera to {camera_id}")
        
    def _on_start_stop(self, event):
        """Handle start/stop button"""
        if not self.running:
            self._start_perception()
        else:
            self._stop_perception()
            
    def _start_perception(self):
        """Start perception processing"""
        if not self.active_cameras:
            wx.MessageBox("No cameras configured. Please configure cameras first.",
                         "Error", wx.OK | wx.ICON_ERROR)
            return
            
        self.running = True
        self.start_btn.SetLabel("⏹ STOP")
        self.start_btn.SetBackgroundColour(wx.Colour(150, 0, 0))
        self.SetStatusText("Running", 0)
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("Perception system started")
        
    def _stop_perception(self):
        """Stop perception processing"""
        self.running = False
        self.start_btn.SetLabel("▶ START")
        self.start_btn.SetBackgroundColour(wx.Colour(0, 120, 0))
        self.SetStatusText("Stopped", 0)
        
        # Stop recording if active
        if self.video_recorder.is_recording:
            self.video_recorder.stop()
            self.record_btn.SetLabel("⏺ RECORD")
            
        logger.info("Perception system stopped")
        
    def _processing_loop(self):
        """Main processing loop (runs in thread)"""
        while self.running:
            try:
                # Get frames from all active cameras
                frames = self.camera_manager.get_all_frames()
                
                if not frames:
                    time.sleep(0.01)
                    continue
                    
                # Process primary camera
                primary_frame = frames.get(self.primary_camera)
                if primary_frame is not None:
                    # Get current settings
                    settings = self.settings.copy()
                    
                    # Update perception engine settings
                    self.perception_engine.detector.confidence_threshold = settings["confidence_threshold"]
                    self.perception_engine.collision_system.set_ego_speed(settings["ego_speed"])
                    
                    # Process frame
                    result_frame, metrics = self.perception_engine.process_frame(
                        primary_frame,
                        enable_detection=settings["enable_detection"],
                        enable_lanes=settings["enable_lanes"],
                        enable_tracking=settings["enable_tracking"]
                    )
                    
                    # Generate bird's eye view if enabled
                    bev_frame = None
                    if settings["show_bev"]:
                        tracked = list(self.perception_engine.tracker.objects.values())
                        bev_frame = self.perception_engine.get_bird_eye_view(primary_frame, tracked)
                    
                    # Record if active
                    if self.video_recorder.is_recording:
                        self.video_recorder.write_frame(result_frame)
                    
                    # Post update event to GUI
                    evt = FrameUpdateEvent(
                        primary_frame=result_frame,
                        secondary_frames={k: v for k, v in frames.items() if k != self.primary_camera},
                        bev_frame=bev_frame
                    )
                    wx.PostEvent(self, evt)
                    
                    # Post metrics update
                    metrics_evt = MetricsUpdateEvent(metrics=metrics)
                    wx.PostEvent(self, metrics_evt)
                    
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
                
    def _on_frame_update(self, event):
        """Handle frame update event"""
        # Update main camera view
        if hasattr(event, 'primary_frame') and event.primary_frame is not None:
            self.main_camera_panel.update_frame(event.primary_frame)
            
        # Update secondary camera views
        if hasattr(event, 'secondary_frames'):
            secondary_ids = list(event.secondary_frames.keys())[:3]
            for i, panel in enumerate(self.secondary_panels):
                if i < len(secondary_ids):
                    panel.update_frame(event.secondary_frames[secondary_ids[i]])
                    
        # Update bird's eye view
        if hasattr(event, 'bev_frame') and event.bev_frame is not None:
            self.bev_panel.update_frame(event.bev_frame)
            
    def _on_metrics_update(self, event):
        """Handle metrics update event"""
        if hasattr(event, 'metrics'):
            self.dashboard.update_metrics(event.metrics)
            
    def _on_setting_changed(self, key: str, value: Any):
        """Handle setting change from control panel"""
        self.settings[key] = value
        logger.debug(f"Setting changed: {key} = {value}")
        
    def _on_record(self, event):
        """Handle record button"""
        if not self.video_recorder.is_recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"adas_recording_{timestamp}.mp4"
            
            # Get frame size from primary camera
            frame = self.camera_manager.get_frame(self.primary_camera)
            if frame is not None:
                h, w = frame.shape[:2]
                self.video_recorder.start(output_path, (w, h))
                self.record_btn.SetLabel("⏹ STOP REC")
                self.record_btn.SetBackgroundColour(wx.Colour(200, 0, 0))
                self.SetStatusText(f"Recording: {output_path}", 2)
        else:
            # Stop recording
            self.video_recorder.stop()
            self.record_btn.SetLabel("⏺ RECORD")
            self.record_btn.SetBackgroundColour(wx.NullColour)
            self.SetStatusText("", 2)
            
    def _on_open_video(self, event):
        """Open video file for processing"""
        with wx.FileDialog(self, "Open Video File", wildcard="Video files (*.mp4;*.avi;*.mkv)|*.mp4;*.avi;*.mkv",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                video_path = dialog.GetPath()
                self._process_video_file(video_path)
                
    def _process_video_file(self, video_path: str):
        """Process video file instead of live camera"""
        # Stop live cameras
        self._stop_perception()
        self.camera_manager.stop_all()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            wx.MessageBox(f"Could not open video: {video_path}", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        self.running = True
        self.start_btn.SetLabel("⏹ STOP")
        self.SetStatusText(f"Playing: {Path(video_path).name}", 0)
        
        def video_loop():
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                    
                settings = self.settings.copy()
                result_frame, metrics = self.perception_engine.process_frame(
                    frame,
                    enable_detection=settings["enable_detection"],
                    enable_lanes=settings["enable_lanes"],
                    enable_tracking=settings["enable_tracking"]
                )
                
                evt = FrameUpdateEvent(primary_frame=result_frame, secondary_frames={}, bev_frame=None)
                wx.PostEvent(self, evt)
                
                metrics_evt = MetricsUpdateEvent(metrics=metrics)
                wx.PostEvent(self, metrics_evt)
                
                # Match video FPS
                time.sleep(1/30)
                
            cap.release()
            
        threading.Thread(target=video_loop, daemon=True).start()
        
    def _on_camera_config(self, event):
        """Show camera configuration dialog"""
        self._stop_perception()
        self._show_camera_dialog()
        
    def _on_about(self, event):
        """Show about dialog"""
        info = wx.adv.AboutDialogInfo()
        info.SetName("ADAS Perception System")
        info.SetVersion("1.0.0")
        info.SetDescription("Advanced Driver Assistance System\nLevel 1+ Autonomous Perception Stack\n\n"
                           "Features:\n"
                           "• Multi-camera support (up to 4)\n"
                           "• Real-time object detection (YOLO)\n"
                           "• Multi-object tracking\n"
                           "• Lane detection\n"
                           "• Collision warning (TTC)\n"
                           "• Bird's eye view\n"
                           "• Video recording")
        info.SetCopyright("(C) 2025 DeepMost AI")
        info.AddDeveloper("DeepMost AI Perception Team")
        wx.adv.AboutBox(info)
        
    def _on_exit(self, event):
        """Exit application"""
        self.Close()
        
    def _on_close(self, event):
        """Handle window close"""
        self.running = False
        self.camera_manager.stop_all()
        
        if self.video_recorder.is_recording:
            self.video_recorder.stop()
            
        self.Destroy()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Check dependencies
    print("=" * 60)
    print("ADAS Perception System - Initializing")
    print("=" * 60)
    
    # Try to import optional dependencies
    try:
        from ultralytics import YOLO
        print("✓ YOLOv8 (ultralytics) available")
    except ImportError:
        print("✗ ultralytics not found - will use OpenCV DNN fallback")
        print("  Install with: pip install ultralytics")
        
    print("✓ OpenCV version:", cv2.__version__)
    print("✓ NumPy version:", np.__version__)
    print("✓ wxPython version:", wx.__version__)
    print("=" * 60)
    
    # Create and run application
    app = wx.App()
    
    # Set app-wide dark theme
    if hasattr(wx, 'SystemOptions'):
        wx.SystemOptions.SetOption("msw.dark-mode", 2)
        
    frame = MainFrame()
    frame.Show()
    frame.Center()
    
    app.MainLoop()


if __name__ == "__main__":
    main()