#!/usr/bin/env python3
"""
Ultra-Advanced Features Module
All advanced detection, tracking, and analysis features
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
from enum import Enum, auto
import logging

logger = logging.getLogger('UltraFeatures')


# ============================================================================
# VEHICLE TYPE CLASSIFIER
# ============================================================================

class VehicleTypeClassifier:
    """Classify vehicle types based on shape and size"""

    def classify(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> str:
        """Classify vehicle type from bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width == 0 or height == 0:
            return "UNKNOWN"

        aspect_ratio = width / height
        area = width * height

        # Extract vehicle region
        vehicle_img = frame[y1:y2, x1:x2]

        if vehicle_img.size == 0:
            return "UNKNOWN"

        # Calculate features
        avg_brightness = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY).mean()

        # Classification rules (simplified)
        if aspect_ratio > 2.5:
            if area > 30000:
                return "BUS"
            else:
                return "TRUCK"
        elif aspect_ratio > 1.8:
            if area > 15000:
                return "SUV"
            else:
                return "SEDAN"
        elif aspect_ratio < 1.2:
            return "MOTORCYCLE"
        else:
            return "CAR"


# ============================================================================
# LICENSE PLATE DETECTOR
# ============================================================================

class LicensePlateDetector:
    """Detect license plates (not OCR)"""

    def __init__(self):
        # Load cascade classifier for license plates
        cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray, vehicle_bbox: Tuple) -> Optional[Tuple]:
        """Detect license plate in vehicle region"""
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_img = frame[y1:y2, x1:x2]

        if vehicle_img.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)

        # Detect plates
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 10)
        )

        if len(plates) > 0:
            # Return first detected plate (adjusted to global coordinates)
            px, py, pw, ph = plates[0]
            return (x1 + px, y1 + py, x1 + px + pw, y1 + py + ph)

        # Fallback: detect rectangular shapes in bottom third
        bottom_third = vehicle_img[int(vehicle_img.shape[0]*0.6):, :]
        edges = cv2.Canny(bottom_third, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            # License plates are typically 2-4 times wider than tall
            if 2.0 < aspect_ratio < 5.0 and w > 30 and h > 10:
                # Adjust to global coordinates
                return (x1 + x, y1 + int(vehicle_img.shape[0]*0.6) + y,
                       x1 + x + w, y1 + int(vehicle_img.shape[0]*0.6) + y + h)

        return None


# ============================================================================
# POTHOLE DETECTOR
# ============================================================================

class PotholeDetector:
    """Detect potholes and road damage"""

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """Detect potholes in road surface"""
        height, width = frame.shape[:2]

        # Focus on road region (bottom 40%)
        road_region = frame[int(height*0.6):, :]

        # Convert to grayscale
        gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect dark circular regions (potholes)
        circles = cv2.HoughCircles(
            filtered,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        potholes = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle
                # Check if region is darker than surroundings
                mask = np.zeros_like(gray)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                masked_region = cv2.bitwise_and(gray, gray, mask=mask)
                avg_inside = masked_region[masked_region > 0].mean() if np.any(masked_region > 0) else 0

                # Get surrounding pixels
                outer_mask = np.zeros_like(gray)
                cv2.circle(outer_mask, (cx, cy), r+20, 255, -1)
                cv2.circle(outer_mask, (cx, cy), r, 0, -1)
                outer_region = cv2.bitwise_and(gray, gray, mask=outer_mask)
                avg_outside = outer_region[outer_region > 0].mean() if np.any(outer_region > 0) else 0

                # Pothole is darker than surroundings
                if avg_inside < avg_outside - 20:
                    # Adjust coordinates to full frame
                    potholes.append((cx, int(height*0.6) + cy, r))

        return potholes


# ============================================================================
# OPTICAL FLOW ANALYZER
# ============================================================================

class OpticalFlowAnalyzer:
    """Analyze optical flow for motion detection"""

    def __init__(self):
        self.prev_gray = None

    def calculate(self, frame: np.ndarray) -> Optional[Dict]:
        """Calculate dense optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate dominant motion
        avg_flow_x = np.mean(flow[..., 0])
        avg_flow_y = np.mean(flow[..., 1])

        self.prev_gray = gray

        return {
            'flow': flow,
            'magnitude': magnitude,
            'angle': angle,
            'dominant_motion': (avg_flow_x, avg_flow_y),
            'avg_magnitude': np.mean(magnitude)
        }

    def visualize(self, flow_data: Dict, frame_shape: Tuple) -> np.ndarray:
        """Create optical flow visualization"""
        if flow_data is None:
            return np.zeros((*frame_shape[:2], 3), dtype=np.uint8)

        magnitude = flow_data['magnitude']
        angle = flow_data['angle']

        # Create HSV image
        hsv = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)

        # Hue represents direction
        hsv[..., 0] = angle * 180 / np.pi / 2

        # Value represents magnitude
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Saturation is constant
        hsv[..., 1] = 255

        # Convert to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return flow_vis


# ============================================================================
# MOTION PREDICTOR
# ============================================================================

class MotionPredictor:
    """Predict future object positions"""

    def predict(self, tracked_object, num_steps: int = 10) -> List[Tuple]:
        """Predict future positions based on velocity"""
        predictions = []

        if not hasattr(tracked_object, 'velocity'):
            return predictions

        vx, vy = tracked_object.velocity
        cx, cy = tracked_object.centroid

        # Simple linear prediction
        for step in range(1, num_steps + 1):
            future_x = cx + vx * step
            future_y = cy + vy * step
            predictions.append((int(future_x), int(future_y)))

        return predictions

    def predict_collision(self, obj1, obj2, time_horizon: float = 3.0) -> float:
        """Predict collision probability between two objects"""
        if not (hasattr(obj1, 'velocity') and hasattr(obj2, 'velocity')):
            return 0.0

        # Get positions and velocities
        p1 = np.array(obj1.centroid)
        p2 = np.array(obj2.centroid)
        v1 = np.array(obj1.velocity)
        v2 = np.array(obj2.velocity)

        # Relative position and velocity
        rel_pos = p2 - p1
        rel_vel = v2 - v1

        # Time to closest approach
        if np.linalg.norm(rel_vel) < 0.1:
            return 0.0

        t_closest = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)

        if t_closest < 0 or t_closest > time_horizon * 30:  # Assuming 30 FPS
            return 0.0

        # Distance at closest approach
        closest_dist = np.linalg.norm(rel_pos + rel_vel * t_closest)

        # Collision probability (inverse exponential of distance)
        collision_prob = np.exp(-closest_dist / 100.0)

        return min(1.0, collision_prob)


# ============================================================================
# SUDDEN MOVEMENT DETECTOR
# ============================================================================

class SuddenMovementDetector:
    """Detect sudden/erratic movements"""

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.movement_history: Dict[int, deque] = {}

    def detect(self, tracked_objects: List) -> List[int]:
        """Detect objects with sudden movement"""
        sudden_movers = []

        for obj in tracked_objects:
            if not hasattr(obj, 'track_id') or not hasattr(obj, 'velocity'):
                continue

            track_id = obj.track_id

            # Initialize history
            if track_id not in self.movement_history:
                self.movement_history[track_id] = deque(maxlen=10)

            # Calculate speed
            speed = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)

            # Add to history
            self.movement_history[track_id].append(speed)

            # Check for sudden change
            if len(self.movement_history[track_id]) >= 3:
                speeds = list(self.movement_history[track_id])
                recent_avg = np.mean(speeds[-3:])
                prev_avg = np.mean(speeds[:-3]) if len(speeds) > 3 else recent_avg

                # Sudden acceleration or deceleration
                if abs(recent_avg - prev_avg) > self.threshold:
                    sudden_movers.append(track_id)

        return sudden_movers


# ============================================================================
# DRIVING BEHAVIOR ANALYZER
# ============================================================================

class DrivingBehaviorAnalyzer:
    """Analyze and classify driving behavior"""

    def __init__(self):
        self.behavior_history = deque(maxlen=300)
        self.metrics = {
            'sudden_movements': 0,
            'lane_departures': 0,
            'close_calls': 0,
            'smooth_driving_score': 100
        }

    def analyze(self, tracked_objects: List, lane_info, collision_warnings: int) -> str:
        """Analyze driving behavior"""

        # Count aggressive maneuvers
        sudden_count = len([obj for obj in tracked_objects
                          if hasattr(obj, 'acceleration') and abs(obj.acceleration) > 5])

        # Update metrics
        if sudden_count > 2:
            self.metrics['sudden_movements'] += 1

        if hasattr(lane_info, 'departure_warning') and lane_info.departure_warning:
            self.metrics['lane_departures'] += 1

        if collision_warnings > 0:
            self.metrics['close_calls'] += 1

        # Calculate smooth driving score
        penalty = (self.metrics['sudden_movements'] +
                  self.metrics['lane_departures'] * 2 +
                  self.metrics['close_calls'] * 3)

        self.metrics['smooth_driving_score'] = max(0, 100 - penalty)

        # Classify behavior
        if self.metrics['smooth_driving_score'] < 60:
            return "AGGRESSIVE"
        elif self.metrics['smooth_driving_score'] > 85:
            return "CAUTIOUS"
        else:
            return "NORMAL"

    def get_score(self) -> int:
        """Get current driving score"""
        return self.metrics['smooth_driving_score']


# ============================================================================
# ENSEMBLE DETECTOR
# ============================================================================

class EnsembleDetector:
    """Combine multiple detectors for better accuracy"""

    def __init__(self, detectors: List):
        self.detectors = detectors
        self.weights = [1.0 / len(detectors)] * len(detectors)

    def detect(self, frame: np.ndarray) -> List:
        """Run all detectors and combine results"""
        all_detections = []

        # Run each detector
        for detector in self.detectors:
            detections = detector.detect(frame)
            all_detections.extend(detections)

        # Remove duplicates using NMS
        if len(all_detections) == 0:
            return []

        # Convert to format for NMS
        boxes = []
        scores = []
        for det in all_detections:
            if hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                boxes.append(det.bbox)
                scores.append(det.confidence)

        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.3,
            nms_threshold=0.4
        )

        # Return filtered detections
        filtered = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            filtered.append(all_detections[idx])

        return filtered


# ============================================================================
# CONFIDENCE CALIBRATOR
# ============================================================================

class ConfidenceCalibrator:
    """Calibrate and improve confidence scores"""

    def __init__(self):
        self.calibration_data = deque(maxlen=1000)

    def calibrate(self, detections: List) -> List:
        """Calibrate confidence scores"""
        for det in detections:
            if not hasattr(det, 'confidence'):
                continue

            # Apply sigmoid transformation for better calibration
            raw_conf = det.confidence
            calibrated = 1.0 / (1.0 + np.exp(-10 * (raw_conf - 0.5)))

            # Adjust based on object size (larger = more confident)
            if hasattr(det, 'bbox'):
                x1, y1, x2, y2 = det.bbox
                area = (x2 - x1) * (y2 - y1)
                size_factor = min(1.2, 0.8 + area / 50000)
                calibrated *= size_factor

            det.confidence = min(1.0, calibrated)

            # Store for future calibration
            self.calibration_data.append((raw_conf, calibrated))

        return detections


# ============================================================================
# SMALL OBJECT DETECTOR
# ============================================================================

class SmallObjectDetector:
    """Specialized detector for small/far objects"""

    def __init__(self):
        self.min_size = 10

    def detect(self, frame: np.ndarray, base_detections: List) -> List:
        """Detect small objects missed by main detector"""
        height, width = frame.shape[:2]

        # Focus on top half where far objects appear
        roi = frame[:height//2, :]

        # Apply contrast enhancement
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l)
        enhanced = cv2.merge([enhanced_l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Detect edges
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_detections = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter by size
            if w < self.min_size or h < self.min_size or w > 100 or h > 100:
                continue

            # Check if not already detected
            already_detected = False
            for det in base_detections:
                if hasattr(det, 'bbox'):
                    dx1, dy1, dx2, dy2 = det.bbox
                    # Check overlap
                    if (x < dx2 and x+w > dx1 and y < dy2 and y+h > dy1):
                        already_detected = True
                        break

            if not already_detected:
                # Create detection
                from dataclasses import dataclass
                @dataclass
                class SmallDetection:
                    bbox: Tuple
                    confidence: float
                    class_id: int
                    class_name: str

                small_detections.append(SmallDetection(
                    bbox=(x, y, x+w, y+h),
                    confidence=0.5,
                    class_id=2,  # Assume car
                    class_name="vehicle_far"
                ))

        return small_detections


# Only for testing
if __name__ == "__main__":
    print("Ultra Features Module Loaded")
    print("Available Classes:")
    print("  - VehicleTypeClassifier")
    print("  - LicensePlateDetector")
    print("  - PotholeDetector")
    print("  - OpticalFlowAnalyzer")
    print("  - MotionPredictor")
    print("  - SuddenMovementDetector")
    print("  - DrivingBehaviorAnalyzer")
    print("  - EnsembleDetector")
    print("  - ConfidenceCalibrator")
    print("  - SmallObjectDetector")
