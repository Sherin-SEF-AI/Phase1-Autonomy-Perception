#!/usr/bin/env python3
"""
Advanced Modules for ADAS Perception System
Includes: Data Logger, Performance Profiler, Analytics Engine, and more
"""

import sqlite3
import pickle
import gzip
import json
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import deque
import numpy as np
import time


# ============================================================================
# DATA LOGGER & PLAYBACK SYSTEM
# ============================================================================

class DataLogger:
    """Advanced data logging system with compression"""

    def __init__(self, output_dir: str = "adas_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_path = self.output_dir / f"session_{self.session_id}.db"

        self.db_conn = None
        self.is_logging = False
        self.frame_count = 0

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        self.db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = self.db_conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY,
                timestamp REAL,
                frame_data BLOB,
                detections_data BLOB,
                metrics_data BLOB
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                event_data TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_info (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_frames INTEGER,
                metadata TEXT
            )
        ''')

        self.db_conn.commit()

    def start_logging(self):
        """Start logging session"""
        self.is_logging = True
        self.frame_count = 0

        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO session_info (session_id, start_time, total_frames, metadata)
            VALUES (?, ?, 0, ?)
        ''', (self.session_id, time.time(), json.dumps({"status": "started"})))
        self.db_conn.commit()

    def log_frame(self, frame: np.ndarray, detections: List, metrics: Dict):
        """Log a single frame with data"""
        if not self.is_logging:
            return

        # Compress frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_compressed = gzip.compress(buffer.tobytes())

        # Serialize detections and metrics
        detections_serialized = gzip.compress(pickle.dumps(detections))
        metrics_serialized = gzip.compress(pickle.dumps(metrics))

        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO frames (frame_id, timestamp, frame_data, detections_data, metrics_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (self.frame_count, time.time(), frame_compressed, detections_serialized, metrics_serialized))

        self.frame_count += 1

        if self.frame_count % 100 == 0:
            self.db_conn.commit()

    def log_event(self, event_type: str, event_data: Dict):
        """Log an event"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, event_data)
            VALUES (?, ?, ?)
        ''', (time.time(), event_type, json.dumps(event_data)))
        self.db_conn.commit()

    def stop_logging(self):
        """Stop logging session"""
        self.is_logging = False

        cursor = self.db_conn.cursor()
        cursor.execute('''
            UPDATE session_info
            SET end_time = ?, total_frames = ?
            WHERE session_id = ?
        ''', (time.time(), self.frame_count, self.session_id))
        self.db_conn.commit()

    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()


class DataPlayback:
    """Playback logged data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_conn = sqlite3.connect(db_path)
        self.total_frames = self._get_total_frames()
        self.current_frame = 0

    def _get_total_frames(self) -> int:
        """Get total number of frames"""
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT total_frames FROM session_info LIMIT 1')
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_frame(self, frame_id: int) -> Tuple[np.ndarray, List, Dict]:
        """Get frame data by ID"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            SELECT frame_data, detections_data, metrics_data
            FROM frames WHERE frame_id = ?
        ''', (frame_id,))

        result = cursor.fetchone()
        if not result:
            return None, None, None

        # Decompress and deserialize
        frame_data = gzip.decompress(result[0])
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        detections = pickle.loads(gzip.decompress(result[1]))
        metrics = pickle.loads(gzip.decompress(result[2]))

        return frame, detections, metrics

    def get_events(self, event_type: str = None) -> List[Dict]:
        """Get logged events"""
        cursor = self.db_conn.cursor()

        if event_type:
            cursor.execute('SELECT * FROM events WHERE event_type = ?', (event_type,))
        else:
            cursor.execute('SELECT * FROM events')

        events = []
        for row in cursor.fetchall():
            events.append({
                'event_id': row[0],
                'timestamp': row[1],
                'event_type': row[2],
                'event_data': json.loads(row[3])
            })

        return events

    def close(self):
        """Close database"""
        if self.db_conn:
            self.db_conn.close()


# ============================================================================
# PERFORMANCE PROFILER
# ============================================================================

class PerformanceProfiler:
    """Real-time performance monitoring and profiling"""

    def __init__(self):
        self.process = psutil.Process()
        self.metrics_history = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'fps': deque(maxlen=100),
            'latency': deque(maxlen=100)
        }
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)

    def update(self, processing_time_ms: float):
        """Update profiler metrics"""
        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.01)
        self.metrics_history['cpu'].append(cpu_percent)

        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.metrics_history['memory'].append(memory_mb)

        # FPS
        current_time = time.time()
        self.frame_times.append(current_time)

        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
            self.metrics_history['fps'].append(fps)

        # Latency
        self.metrics_history['latency'].append(processing_time_ms)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}

        for key, values in self.metrics_history.items():
            if len(values) > 0:
                stats[key] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
            else:
                stats[key] = {
                    'current': 0,
                    'mean': 0,
                    'max': 0,
                    'min': 0,
                    'std': 0
                }

        stats['uptime'] = time.time() - self.start_time
        return stats

    def get_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        if len(self.metrics_history['cpu']) > 0:
            avg_cpu = np.mean(self.metrics_history['cpu'])
            if avg_cpu > 80:
                bottlenecks.append(f"High CPU usage: {avg_cpu:.1f}%")

        if len(self.metrics_history['memory']) > 0:
            avg_mem = np.mean(self.metrics_history['memory'])
            if avg_mem > 2000:  # 2GB
                bottlenecks.append(f"High memory usage: {avg_mem:.0f}MB")

        if len(self.metrics_history['fps']) > 0:
            avg_fps = np.mean(self.metrics_history['fps'])
            if avg_fps < 20:
                bottlenecks.append(f"Low FPS: {avg_fps:.1f}")

        if len(self.metrics_history['latency']) > 0:
            avg_latency = np.mean(self.metrics_history['latency'])
            if avg_latency > 50:
                bottlenecks.append(f"High latency: {avg_latency:.1f}ms")

        return bottlenecks


# ============================================================================
# DRIVING ANALYTICS ENGINE
# ============================================================================

class DrivingAnalyticsEngine:
    """Advanced driving behavior and scene analytics"""

    def __init__(self):
        self.behavior_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.lane_departure_count = 0
        self.collision_warning_count = 0
        self.aggressive_events = 0
        self.total_distance = 0.0
        self.session_start = time.time()

    def analyze_behavior(self, metrics: Dict, lane_info: Any, tracked_objects: List) -> Dict:
        """Analyze driving behavior"""
        behavior = {
            'type': 'NORMAL',
            'score': 100,
            'events': []
        }

        # Check lane departure
        if hasattr(lane_info, 'departure_warning') and lane_info.departure_warning:
            self.lane_departure_count += 1
            behavior['events'].append('LANE_DEPARTURE')
            behavior['score'] -= 10

        # Check collision warnings
        if metrics.get('collision_risk', 'NONE') in ['DANGER', 'CRITICAL']:
            self.collision_warning_count += 1
            behavior['events'].append('COLLISION_WARNING')
            behavior['score'] -= 20

        # Check for aggressive behavior (rapid acceleration/deceleration)
        rapid_changes = 0
        for obj in tracked_objects:
            if hasattr(obj, 'acceleration') and abs(obj.acceleration) > 5:
                rapid_changes += 1

        if rapid_changes > 2:
            self.aggressive_events += 1
            behavior['events'].append('AGGRESSIVE_BEHAVIOR')
            behavior['score'] -= 15

        # Determine behavior type
        if behavior['score'] < 50:
            behavior['type'] = 'AGGRESSIVE'
        elif behavior['score'] < 70:
            behavior['type'] = 'CAUTIOUS'
        elif len(behavior['events']) > 0:
            behavior['type'] = 'WARNING'

        self.behavior_history.append(behavior)
        return behavior

    def get_session_summary(self) -> Dict:
        """Get driving session summary"""
        session_duration = time.time() - self.session_start

        return {
            'duration_seconds': session_duration,
            'duration_formatted': time.strftime('%H:%M:%S', time.gmtime(session_duration)),
            'lane_departures': self.lane_departure_count,
            'collision_warnings': self.collision_warning_count,
            'aggressive_events': self.aggressive_events,
            'total_distance_km': self.total_distance / 1000,
            'average_behavior_score': np.mean([b['score'] for b in self.behavior_history]) if self.behavior_history else 100
        }

    def get_recommendations(self) -> List[str]:
        """Get driving recommendations"""
        recommendations = []

        if self.lane_departure_count > 5:
            recommendations.append("⚠ Multiple lane departures detected. Stay centered in your lane.")

        if self.collision_warning_count > 3:
            recommendations.append("⚠ Frequent collision warnings. Maintain safe following distance.")

        if self.aggressive_events > 2:
            recommendations.append("⚠ Aggressive driving detected. Maintain smooth acceleration and braking.")

        if not recommendations:
            recommendations.append("✓ Excellent driving! No issues detected.")

        return recommendations


# ============================================================================
# HEATMAP GENERATOR
# ============================================================================

class HeatMapGenerator:
    """Generate attention and detection heatmaps"""

    def __init__(self, size: Tuple[int, int] = (720, 1280)):
        self.height, self.width = size
        self.detection_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.attention_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.decay_rate = 0.98

    def update(self, detections: List, frame_shape: Tuple[int, int]):
        """Update heatmaps with new detections"""
        h, w = frame_shape[:2]

        # Decay existing maps
        self.detection_map *= self.decay_rate
        self.attention_map *= self.decay_rate

        # Resize if needed
        if (h, w) != (self.height, self.width):
            self.detection_map = cv2.resize(self.detection_map, (w, h))
            self.attention_map = cv2.resize(self.attention_map, (w, h))
            self.height, self.width = h, w

        # Add detections to map
        for det in detections:
            if hasattr(det, 'bbox'):
                x1, y1, x2, y2 = det.bbox
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # Weight by confidence and proximity
                weight = det.confidence if hasattr(det, 'confidence') else 1.0

                # Add to detection map
                self.detection_map[y1:y2, x1:x2] += weight * 0.5

                # Add Gaussian blob to attention map
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self._add_gaussian_blob(center_x, center_y, weight)

        # Normalize
        if self.detection_map.max() > 0:
            self.detection_map = np.clip(self.detection_map / self.detection_map.max(), 0, 1)

        if self.attention_map.max() > 0:
            self.attention_map = np.clip(self.attention_map / self.attention_map.max(), 0, 1)

    def _add_gaussian_blob(self, cx: int, cy: int, weight: float, sigma: int = 50):
        """Add Gaussian blob to attention map"""
        y, x = np.ogrid[:self.height, :self.width]
        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        self.attention_map += gaussian * weight * 0.3

    def get_detection_heatmap(self) -> np.ndarray:
        """Get colored detection heatmap"""
        heatmap = (self.detection_map * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    def get_attention_heatmap(self) -> np.ndarray:
        """Get colored attention heatmap"""
        heatmap = (self.attention_map * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)


# ============================================================================
# TRAFFIC SIGN CLASSIFIER
# ============================================================================

class TrafficSignClassifier:
    """Classify detected traffic signs"""

    SIGN_TYPES = {
        'STOP': 0,
        'YIELD': 1,
        'SPEED_LIMIT_30': 2,
        'SPEED_LIMIT_50': 3,
        'SPEED_LIMIT_70': 4,
        'NO_ENTRY': 5,
        'NO_PARKING': 6,
        'ONE_WAY': 7,
        'PEDESTRIAN_CROSSING': 8,
        'SCHOOL_ZONE': 9
    }

    def __init__(self):
        self.model = None

    def classify(self, sign_image: np.ndarray) -> Tuple[str, float]:
        """Classify traffic sign"""
        # Simple shape and color-based classification
        if sign_image is None or sign_image.size == 0:
            return "UNKNOWN", 0.0

        # Resize
        sign_resized = cv2.resize(sign_image, (64, 64))

        # Convert to HSV
        hsv = cv2.cvtColor(sign_resized, cv2.COLOR_BGR2HSV)

        # Detect red (STOP, YIELD, NO_ENTRY)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 | red_mask2

        red_ratio = np.sum(red_mask > 0) / (64 * 64)

        if red_ratio > 0.3:
            # Check shape for octagon (STOP)
            gray = cv2.cvtColor(sign_resized, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

                if len(approx) == 8:
                    return "STOP", 0.85

            return "NO_ENTRY", 0.75

        # Detect blue (info signs)
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / (64 * 64)

        if blue_ratio > 0.3:
            return "INFO_SIGN", 0.70

        return "UNKNOWN", 0.0


# Only include this if called directly (for testing)
if __name__ == "__main__":
    print("Advanced ADAS Modules Loaded Successfully")
    print("=" * 60)
    print("Available Components:")
    print("  - DataLogger: Advanced logging with compression")
    print("  - DataPlayback: Replay logged sessions")
    print("  - PerformanceProfiler: Real-time performance monitoring")
    print("  - DrivingAnalyticsEngine: Behavior analysis")
    print("  - HeatMapGenerator: Attention heatmaps")
    print("  - TrafficSignClassifier: Sign recognition")
    print("=" * 60)
